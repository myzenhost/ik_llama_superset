#include "offload_attention.h"   // S69 #220
#include "offload_attention_dense.h"   // dense-layer set (N | -N | list)
#include "llama-kv-xattn-op.h"   // XAttention prefill block selection (FA path)
#include "llama-build-context.h"
#include "llama-turbo-graph.h"
#include "llama-hparams.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-moe-cache.h"
#include "llama-delta-net.h"

#include "ggml.h"

#include <unordered_set>
#include <algorithm>

uint32_t llm_build_context::llama_kv_qnext_state_slots(const llama_kv_cache_fork & kv_self) {
    uint32_t n_slots = 0;

    for (const ggml_tensor * t : kv_self.s_l) {
        if (t == nullptr) {
            continue;
        }

        const uint32_t layer_slots = (uint32_t) t->ne[1];
        if (n_slots == 0) {
            n_slots = layer_slots;
        } else {
            GGML_ASSERT(n_slots == layer_slots);
        }
    }

    return n_slots;
}

std::vector<llm_build_context::recurrent_run> llm_build_context::recurrent_runs() const {
    std::vector<recurrent_run> runs;
    const uint32_t n_slots = llama_kv_qnext_state_slots(kv_self);
    // reserve-graph builds carry no sequence info: one run on slot 0, as before
    const bool has_seq = batch.n_seq_id != nullptr && batch.seq_id != nullptr;
    std::unordered_set<llama_seq_id> seen;
    for (int32_t i = 0; i < n_tokens; ++i) {
        llama_seq_id s = 0;
        if (has_seq) {
            GGML_ASSERT(batch.n_seq_id[i] == 1 && "hybrid recurrent state: a token must belong to exactly one sequence");
            s = batch.seq_id[i][0];
        }
        if (runs.empty() || runs.back().seq != s) {
            // each run reads its state row and writes it back inside the graph; a second run of the
            // same sequence in one ubatch would read that row before the first run's write lands
            GGML_ASSERT(seen.insert(s).second && "hybrid recurrent state: a sequence's tokens must be contiguous in the ubatch");
            GGML_ASSERT(s >= 0 && (n_slots == 0 || (uint32_t) s < n_slots));
            runs.push_back({ i, 0, s, batch.pos != nullptr && batch.pos[i] == 0 });
        }
        runs.back().len++;
    }
    return runs;
}

llm_build_context::llm_build_context(
        llama_context  & lctx,
    const llama_batch  & batch,
    const llm_build_cb & cb,
    bool   worst_case,
    bool   warmup,
    int    n_outputs_,
    bool   clear_lctx_inputs,
    std::vector<uint8_t> * buf_compute_meta_override) :
        model            (lctx.model),
        lctx             (lctx),
        hparams          (model.hparams),
        cparams          (lctx.cparams),
        batch            (batch),
        kv_self          (lctx.kv_self),
        n_embd           (hparams.n_embd),
        n_layer          (hparams.n_layer),
        n_rot            (hparams.n_rot),
        n_ctx            (cparams.n_ctx),
        n_head           (hparams.n_head()),
        n_head_kv        (hparams.n_head_kv()),
        n_embd_head_k    (hparams.n_embd_head_k(0)),
        n_embd_k_gqa     (hparams.n_embd_k_gqa()),
        n_embd_head_v    (hparams.n_embd_head_v(0)),
        n_embd_v_gqa     (hparams.n_embd_v_gqa()),
        n_expert         (hparams.n_expert),
        n_expert_used    (warmup ? hparams.n_expert : hparams.n_expert_used),
        freq_base        (cparams.rope_freq_base),
        freq_scale       (cparams.rope_freq_scale),
        ext_factor       (cparams.yarn_ext_factor),
        attn_factor      (cparams.yarn_attn_factor),
        beta_fast        (cparams.yarn_beta_fast),
        beta_slow        (cparams.yarn_beta_slow),
        norm_eps         (hparams.f_norm_eps),
        norm_rms_eps     (hparams.f_norm_rms_eps),
        n_tokens         (batch.n_tokens),
        // legacy mamba scans ONE sequence per ubatch (build_mamba.cpp asserts n_kv == 1; llama_decode_internal cuts
        // ubatches at sequence boundaries for it), so one state slot IS its largest graph. kv_self.size is n_seq_max
        // and made the constructor-time reserve abort for every n_seq_max > 1.
        n_kv             (worst_case ? (lctx.model.arch == LLM_ARCH_MAMBA ? 1 : kv_self.size) : kv_self.n),
        // Pooled embeddings (embeddings on, pooling not NONE) make EVERY token an output: llama_decode_internal counts
        // them all (its `embd_pooled`), and append_pooling multiplies the [n_embd, n_outputs] hidden states by an
        // [n_tokens, n_tokens] matrix for MEAN and ATTENTION pooling. A reserve graph with fewer outputs than tokens
        // (the pruned-output graph of the legacy reserve, or --worst-graph-tokens) is therefore a shape that cannot
        // occur, and it tripped GGML_ASSERT(ggml_can_mul_mat) in the constructor-time reserve: MEASURED 2026-11-13,
        // `llama-server --embedding --pooling mean` died at load on stories260K and stories15M_MOE (rc 0xC0000409),
        // while --pooling last / none loaded. So a worst-case graph for pooled embeddings is always all-outputs.
        n_outputs        (worst_case ? ((n_outputs_ > 0 &&
                                         !(lctx.cparams.embeddings && lctx.cparams.pooling_type != LLAMA_POOLING_TYPE_NONE))
                                        ? n_outputs_ : n_tokens)
                                     : lctx.n_outputs),
        n_outputs_enc    (worst_case ? n_tokens : lctx.embd_enc.size() / hparams.n_embd),
        kv_head          (worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head),
        swa_head         (!kv_self.any_compacted() ? kv_head :
                          worst_case ? (int32_t) (kv_self.size_swa - n_tokens) : (int32_t) kv_self.head_swa),
        n_ctx_orig       (cparams.n_ctx_orig_yarn),
        flash_attn       (cparams.flash_attn),
        mla_attn         (cparams.mla_attn),
        attn_max_batch   (cparams.attn_max_batch),
        fused_moe_up_gate(cparams.fused_moe_up_gate),
        grouped_expert_routing(cparams.grouped_expert_routing),
        fused_up_gate    (cparams.fused_up_gate),
        fused_mmad       (cparams.fused_mmad),
        rope_cache       (cparams.rope_cache),
        k_cache_hadamard (cparams.k_cache_hadamard),
        split_mode_graph_scheduling (cparams.split_mode_graph_scheduling),
        min_experts      (cparams.min_experts),
        thresh_experts   (cparams.thresh_experts),
        pooling_type     (cparams.pooling_type),
        rope_type        (hparams.rope_type),
        clear_lctx_inputs(clear_lctx_inputs),
        cb               (cb),
        buf_compute_meta (buf_compute_meta_override ? *buf_compute_meta_override : lctx.buf_compute_meta) {
            // all initializations should be done in init()
}

void llm_build_context::init() {
    // a new graph is being built: drop the previous graph's rotated-activation memo
    // (the cached tensors live in the previous graph's context)
    lctx.hadamard_memo.clear();

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx0 = ggml_init(params);

    // YaRN truncate=false models (HF rope_parameters["truncate"] == false): every YaRN rope node this graph
    // creates carries the flag; the ~470 ggml_rope_ext call sites stay untouched.
    ggml_set_rope_yarn_truncate(ctx0, model.hparams.rope_yarn_truncate);

    if (clear_lctx_inputs) {
        lctx.lora_seq_masks.clear();
        lctx.lora_gate_masks.clear();
        lctx.lora_seq_n_tokens = n_tokens;
        lctx.inp_lily_seq_mean = nullptr;
        lctx.inp_tokens      = nullptr;
        lctx.inp_embd        = nullptr;
        lctx.inp_pos         = nullptr;
        lctx.inp_ffd_idx     = nullptr;
        lctx.inp_out_ids     = nullptr;
        lctx.inp_KQ_mask     = nullptr;
        lctx.inp_KQ_mask_swa = nullptr;
        lctx.inp_KQ_mask_swa_win = nullptr;
        lctx.inp_dg_sc_logits    = nullptr;
        lctx.swa_window_view = {};
        lctx.inp_K_shift     = nullptr;
        lctx.inp_mean        = nullptr;
        lctx.inp_cls         = nullptr;
        lctx.inp_s_copy      = nullptr;
        lctx.inp_s_mask      = nullptr;
        lctx.inp_s_seq       = nullptr;
        lctx.inp_s_seq_qnext = nullptr;
        lctx.inp_inkling_tau         = nullptr;
        lctx.inp_inkling_rel_idx     = nullptr;
        lctx.inp_inkling_rel_idx_swa = nullptr;
        lctx.inp_inkling_vocab_mask  = nullptr;
        lctx.inp_inkling_shexp_idx   = nullptr;
        lctx.inp_pos_bucket    = nullptr;
        lctx.inp_embd_enc      = nullptr;
        lctx.inp_KQ_mask_cross = nullptr;
        lctx.inp_dsa_sink      = nullptr;
        lctx.inp_kpool_cells     = nullptr;
        lctx.inp_kpool_bias      = nullptr;
        lctx.inp_kpool_tail      = nullptr;
        lctx.inp_kpool_ape_slots = nullptr;
        lctx.inp_mtp_carry     = nullptr;
        lctx.inp_qsa.clear();
        lctx.dflash.inputs.target_features = nullptr;
        lctx.dflash.inputs.pos_ctx = nullptr;
        lctx.dflash.inputs.kq_mask = nullptr;
        lctx.dsv4.inputs.raw_k_write_src_idxs = nullptr;
        lctx.dsv4.inputs.raw_k_write_idxs = nullptr;
        lctx.dsv4.inputs.raw_k_read_idxs = nullptr;
        lctx.dsv4.inputs.csa = {};
        lctx.dsv4.inputs.hca = {};
        lctx.dsv4.inputs.lid = {};
    }
}

void llm_build_context::free() {
    if (ctx0) {
        ggml_free(ctx0);
        ctx0 = nullptr;
    }
}

bool llm_build_context::supports_op(const ggml_tensor * op) const {
    for (ggml_backend_t backend : lctx.backends) {
        if (ggml_backend_supports_op(backend, op)) {
            return true;
        }
    }
    return false;
}

ggml_cgraph * llm_build_context::build_k_shift() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    GGML_ASSERT(kv_self.size == n_ctx);

    const auto & rope_type_shift = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
        // @ngxson : this is a workaround
        // for M-RoPE, we want to rotate the whole vector when doing KV shift
        // a normal RoPE should work, we just need to use the correct ordering
        // ref: https://github.com/ggml-org/llama.cpp/pull/13870
        ? LLAMA_ROPE_TYPE_NEOX
        : hparams.rope_type;

    const float yarn_attn_factor_shift = model.arch == LLM_ARCH_DEEPSEEK2 || llm_arch_is_dsv4(model.arch) || model.arch == LLM_ARCH_MISTRAL4
        ? 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale))
        : cparams.yarn_attn_factor;

    lctx.inp_K_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx);
    cb(lctx.inp_K_shift, "K_shift", -1);
    ggml_set_input(lctx.inp_K_shift);

    const int n_kv_layers = model.mtp ? hparams.n_layer : hparams.n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < n_kv_layers; ++il) {
        if (llm_arch_is_hybrid(model.arch) && hparams.is_recurrent(il)) {
            continue;
        }
        if (kv_self.k_l[il] == nullptr) {
            continue;
        }
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_embd_head_k_l = hparams.n_embd_head_k(il);
        const int     n_rot_l = hparams.rope_n_rot(il);
        float freq_base_l  = hparams.is_swa_impl[il] ? hparams.rope_freq_base_train_swa  : freq_base;
        float freq_scale_l = hparams.is_swa_impl[il] ? hparams.rope_freq_scale_train_swa : freq_scale;
        if (hparams.has_rope_freq_base_per_layer) {
            freq_base_l = hparams.rope_freq_base_per_layer[il];
        }
        struct ggml_tensor * rope_factors = nullptr;
        const uint32_t apply_mask = hparams.rope_scaling_apply_mask;
        if ((hparams.is_swa_impl[il] && (apply_mask & 0x2)) || (!hparams.is_swa_impl[il] && (apply_mask & 0x1))) {
            rope_factors = build_rope_factors(il);
        }
        struct ggml_tensor * k =
            ggml_view_3d(ctx0, kv_self.k_l[il],
                    n_embd_head_k_l, n_head_kv, n_ctx,
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k_l),
                    n_head_kv*ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k_l),
                    0);

        struct ggml_tensor * tmp;
        if (ggml_is_quantized(k->type)) {
            // dequantize to f32 -> RoPE -> quantize back
            tmp = ggml_cast(ctx0, k, GGML_TYPE_F32);
            cb(tmp, "K_f32", il);
            for (auto * backend : lctx.backends) {
                // Figure out which backend KV cache belongs to
                if (ggml_backend_supports_buft(backend, lctx.model.buft_layer[il].buft)) {
                    ggml_backend_sched_set_tensor_backend(lctx.sched, tmp, backend);
                    break;
                }
            }
            tmp = ggml_rope_ext_inplace(ctx0, tmp,
                    lctx.inp_K_shift, rope_factors, n_rot_l, rope_type_shift, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, yarn_attn_factor_shift, beta_fast, beta_slow);
            cb(tmp, "K_shifted_f32", il);
            tmp = ggml_cpy(ctx0, tmp, k);
        } else {
            // we rotate only the first n_rot dimensions
            tmp = ggml_rope_ext_inplace(ctx0, k,
                    lctx.inp_K_shift, rope_factors, n_rot_l, rope_type_shift, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, yarn_attn_factor_shift, beta_fast, beta_slow);
        }
        cb(tmp, "K_shifted", il);
        ggml_build_forward_expand(gf, tmp);
    }

    // DSA lightning-indexer key cache (GLM-5.2 / DeepSeek-V3.2): the persistent indexer keys in
    // kv_self.kr_l[il] are RoPE-position-encoded at write time (build_deepseek2_dsa_indexer rotates
    // the first rope_dim dims with the cell's absolute pos), so a context-shift that re-RoPEs the
    // main K cache MUST apply the identical delta-rotation to the indexer keys, or query/key
    // rotations desync and the indexer top-k silently degrades. We mirror the main K-shift here.
    //
    // Subtlety: the cached key is H * concat(RoPE(k_pe,pos), k_nope), where H is the symmetric
    // orthonormal Walsh-Hadamard rotation applied AFTER RoPE (head-wide, mixing pe+nope dims). So we
    // cannot RoPE the cached key directly. We un-rotate by H (H==H^T, H*H==I), RoPE-delta the pe
    // sub-block, then re-rotate by H. This is exact because (a) for this model ext_factor==0 and
    // attn_factor==1 (no YaRN: GLM-DSA carries no rope-scaling metadata), so NEOX RoPE is a pure
    // rotation and composable: RoPE(x,pos+delta)==RoPE(RoPE(x,pos),delta); and (b) when the Hadamard
    // is disabled (DSA_HADAMARD_DISABLE) the un/re-rotate by H is simply absent (kr_had==false).
    //
    // NOTE (engine gate): build_k_shift only runs after get_can_shift() passes, which currently
    // returns FALSE for ALL MLA models (llama.cpp llama_kv_cache_update_internal -> get_can_shift:
    // is_mla_model() includes GLM_DSA). So on GLM-5.2 a RoPE context-shift is refused by the engine
    // (decode returns 1 == "failed to eval") BEFORE any kr_l graph is built, identically with or
    // without the indexer. This block is therefore correct-and-dormant: it never executes on the
    // current MLA path, but keeps the indexer keys consistent the instant MLA K-shift is ever enabled.
    // It is NOT a way for the indexer to desync — the shift simply does not happen for MLA.
    if (model.arch == LLM_ARCH_GLM_DSA && hparams.indexer_head_size > 0 && !kv_self.kr_l.empty()) {
        const int64_t head_size = hparams.indexer_head_size;
        const int64_t rope_dim  = n_rot;              // n_embd_head_qk_rope (indexer pe dims)
        const int64_t nope_dim  = head_size - rope_dim;

        // Hadamard size used by the forward indexer: largest power-of-2 divisor of head_size that
        // spans the full head row (else the forward path skips it). Rebuild the same matrix here so
        // we can un/re-rotate the cached key. Must match build_deepseek2_dsa_indexer exactly.
        bool do_hadamard = false;
        static const bool dsa_had_disable = getenv("DSA_HADAMARD_DISABLE") != nullptr;
        if (lctx.cparams.dsa_indexer_hadamard && !dsa_had_disable) {
            GGML_ASSERT((head_size & ~(head_size-1)) == head_size);
            do_hadamard = true;
        }
        int64_t nrot = 1;
        while ((nrot * 2) <= head_size && head_size % (nrot * 2) == 0) nrot *= 2;

        for (int il = 0; il < n_layer; ++il) {
            if ((size_t) il >= kv_self.kr_l.size() || kv_self.kr_l[il] == nullptr) {
                continue;
            }
            ggml_tensor * kr = kv_self.kr_l[il]; // {head_size, kv_size} F16
            // The indexer forward RoPE (build_deepseek2_dsa_indexer) passes rope_factors==nullptr;
            // mirror that exactly here so the delta-rotation matches the encoding.
            struct ggml_tensor * rope_factors = nullptr;

            // {head_size, kv_size} -> work in F32 for the rotation, write back F16.
            ggml_tensor * kr_f32 = ggml_cast(ctx0, kr, GGML_TYPE_F32);
            for (auto * backend : lctx.backends) {
                if (ggml_backend_supports_buft(backend, lctx.model.buft_layer[il].buft)) {
                    ggml_backend_sched_set_tensor_backend(lctx.sched, kr_f32, backend);
                    break;
                }
            }
            cb(kr_f32, "kr_f32", il);

            // un-Hadamard: H * kr  ==  concat(RoPE(k_pe,pos), k_nope)  (H symmetric/orthonormal).
            if (do_hadamard) {
                kr_f32 = ggml_hadamard(ctx0, kr_f32, head_size);
                cb(kr_f32, "kr_unhad", il);
            }

            // view the pe sub-block {rope_dim, 1, kv_size} and the nope sub-block {nope_dim,1,kv_size}.
            ggml_tensor * kr_pe = ggml_view_3d(ctx0, kr_f32, rope_dim, 1, n_ctx,
                    ggml_row_size(kr_f32->type, head_size),
                    ggml_row_size(kr_f32->type, head_size), 0);
            ggml_tensor * kr_nope = ggml_view_3d(ctx0, kr_f32, nope_dim, 1, n_ctx,
                    ggml_row_size(kr_f32->type, head_size),
                    ggml_row_size(kr_f32->type, head_size),
                    ggml_row_size(kr_f32->type, rope_dim));
            // RoPE the pe block by the per-cell delta (inp_K_shift holds cells[i].delta) into a NEW
            // tensor (non-in-place; ggml_cont copies the view first to avoid aliasing the source).
            // NEOX, identical params to the indexer forward RoPE.
            ggml_tensor * kr_pe_rot = ggml_rope_ext(ctx0, ggml_cont(ctx0, kr_pe),
                    lctx.inp_K_shift, rope_factors, n_rot, LLAMA_ROPE_TYPE_NEOX, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(kr_pe_rot, "kr_pe_shifted", il);

            // reassemble {head_size, 1, kv_size} = concat(rotated pe, untouched nope) along dim0.
            ggml_tensor * kr_cat = ggml_concat(ctx0, kr_pe_rot, ggml_cont(ctx0, kr_nope), 0);
            kr_cat = ggml_reshape_2d(ctx0, kr_cat, head_size, n_ctx);
            cb(kr_cat, "kr_cat_shifted", il);

            // re-Hadamard: H * concat(RoPE(k_pe,pos+delta), k_nope) == the new cached key.
            ggml_tensor * kr_new = kr_cat;
            if (do_hadamard) {
                kr_new = ggml_hadamard(ctx0, kr_cat, head_size);
                cb(kr_new, "kr_rehad", il);
            }
            // write back into the F16 cache.
            ggml_tensor * kr_back = ggml_cpy(ctx0, kr_new, kr);
            cb(kr_back, "kr_shifted", il);
            ggml_build_forward_expand(gf, kr_back);
        }
    }

    return gf;
}

ggml_cgraph * llm_build_context::build_s_copy() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const uint32_t qnext_state_slots = llama_kv_qnext_state_slots(kv_self);
    const bool has_qnext_state = qnext_state_slots > 0;
    GGML_ASSERT(kv_self.recurrent || has_qnext_state);

    struct ggml_tensor * state_copy = build_inp_s_copy();

    for (int il = 0; il < n_layer; ++il) {
        if (kv_self.recurrent) {
            struct ggml_tensor * conv_states = ggml_reshape_2d(ctx0, kv_self.k_l[il], hparams.n_embd_k_s(), kv_self.size);
            struct ggml_tensor * ssm_states  = ggml_reshape_2d(ctx0, kv_self.v_l[il], hparams.n_embd_v_s(), kv_self.size);

            conv_states = ggml_get_rows(ctx0, conv_states, state_copy);
            ssm_states  = ggml_get_rows(ctx0,  ssm_states, state_copy);

            // TODO: name the intermediate tensors with cb()

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_states, kv_self.k_l[il]));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0,  ssm_states, kv_self.v_l[il]));
        }

        if (kv_self.s_l.size() > (size_t) il && kv_self.s_l[il] != nullptr) {
            // the row is the layer's whole state row (recurrent_state_layout_for: conv + ssm + PLE / Inkling histories, KDA layout),
            // not n_embd_v_s() -- that narrower width aborted this reshape on a seq_cp in a unified cache (test-llama-context "shared")
            const int64_t qnext_row = ggml_nelements(kv_self.s_l[il]) / kv_self.s_l[il]->ne[1];
            struct ggml_tensor * qnext_states_all = ggml_reshape_2d(ctx0, kv_self.s_l[il], qnext_row, kv_self.s_l[il]->ne[1]);
            GGML_ASSERT((uint32_t) qnext_states_all->ne[1] == qnext_state_slots);
            struct ggml_tensor * qnext_state_copy = ggml_view_1d(ctx0, state_copy, qnext_state_slots, 0);
            struct ggml_tensor * qnext_states = ggml_get_rows(ctx0, qnext_states_all, qnext_state_copy);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, qnext_states, kv_self.s_l[il]));
        }
    }

    return gf;
}

ggml_cgraph * llm_build_context::build_defrag(const std::vector<uint32_t> & ids) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    for (uint32_t i = 0; i < ids.size(); ++i) {
        const uint32_t id = ids[i];

        if (i == id || id == ids.size()) {
            continue;
        }

        uint32_t nm = 1;

        while (i + nm < ids.size() && ids[i + nm] == id + nm) {
            nm++;
        }

        // Bound by the cache, not by hparams.n_layer: without an MTP context the cache holds
        // n_layer - nextn_predict_layers layers (llama.cpp llama_kv_cache_init), so a model with a
        // nextn head (qwen35) read k_l[k_l.size()] past the end here and faulted on whatever that
        // slot held -- 0xC0000005 reading FFFFFFFFFFFFFFFF on the first defrag after a -np 4
        // batch (MEASURED 2026-09-26). build_k_shift already bounds its loop the same way.
        const int n_kv_layers = std::min<int>(n_layer, (int) kv_self.k_l.size());
        for (int il = 0; il < n_kv_layers; ++il) {
            if (llm_arch_is_hybrid(model.arch) && hparams.is_recurrent(il)) {
                continue;
            }
            if (kv_self.k_l[il] == nullptr) {
                continue;
            }
            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            ggml_tensor * view_k_src = ggml_view_2d(ctx0, kv_self.k_l[il],
                    n_embd_k_gqa, nm,
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa*i));

            ggml_tensor * view_k_dst = ggml_view_2d(ctx0, kv_self.k_l[il],
                    n_embd_k_gqa, nm,
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa*id));

            ggml_tensor * view_v_src = nullptr;
            ggml_tensor * view_v_dst = nullptr;

            if (kv_self.v_l.size() > il && kv_self.v_l[il] != nullptr) {
                // Note: with MLA the V cache may not be present.
                if (flash_attn) {
                    // NOTE: the V cache is not transposed when using flash attention
                    view_v_src = ggml_view_2d(ctx0, kv_self.v_l[il],
                            n_embd_v_gqa, nm,
                            ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa),
                            ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa*i));

                    view_v_dst = ggml_view_2d(ctx0, kv_self.v_l[il],
                            n_embd_v_gqa, nm,
                            ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa),
                            ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa*id));
                } else {
                    view_v_src = ggml_view_2d(ctx0, kv_self.v_l[il],
                            nm, n_embd_v_gqa,
                            ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                            ggml_row_size(kv_self.v_l[il]->type, i));

                    view_v_dst = ggml_view_2d(ctx0, kv_self.v_l[il],
                            nm, n_embd_v_gqa,
                            ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                            ggml_row_size(kv_self.v_l[il]->type, id));
                }
            }

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_k_src, view_k_dst));
            if (view_v_src && view_v_dst) {
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_v_src, view_v_dst));
            }

            // DSA lightning-indexer key cache: move the indexer keys alongside k_l. Each cell's
            // indexer key (kr_l row, one per cache cell) must follow its cell, or after defrag the
            // kr_l rows no longer match the cells the indexer scores by cell index. The indexer key
            // is RoPE-encoded at its (unchanged) absolute pos, and defrag does NOT change pos, so a
            // plain row move is correct (no re-RoPE needed; only seq_add/K-shift change pos).
            if ((size_t) il < kv_self.kr_l.size() && kv_self.kr_l[il] != nullptr) {
                const int64_t head_size = hparams.indexer_head_size;
                ggml_tensor * view_kr_src = ggml_view_2d(ctx0, kv_self.kr_l[il],
                        head_size, nm,
                        ggml_row_size(kv_self.kr_l[il]->type, head_size),
                        ggml_row_size(kv_self.kr_l[il]->type, head_size*i));
                ggml_tensor * view_kr_dst = ggml_view_2d(ctx0, kv_self.kr_l[il],
                        head_size, nm,
                        ggml_row_size(kv_self.kr_l[il]->type, head_size),
                        ggml_row_size(kv_self.kr_l[il]->type, head_size*id));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_kr_src, view_kr_dst));
            }
        }

        i += nm - 1;
    }

    //LLAMA_LOG_INFO("gf->n_nodes = %d\n", gf->n_nodes);

    return gf;
}

// defined below; declared here for use in the MTP embedding path
static struct ggml_tensor * llm_build_hadamard_rotate(
        struct ggml_context * ctx0,
        struct ggml_tensor * cur,
        const llama_hadamard_transform & t,
        bool inverse);

struct ggml_tensor * llm_build_context::build_inp_embd_mtp(struct ggml_tensor * mtp_tok_embd) {
    struct ggml_tensor * cur = nullptr;

    lctx.inp_engram_gate_mask = nullptr;
    lctx.inp_engram_rows.clear();
    lctx.inp_engram_gate_ids.clear();
    if (batch.token) {
        lctx.inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch.n_tokens);

        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);

        cur = ggml_get_rows(ctx0, mtp_tok_embd, lctx.inp_tokens);

        // undo the Hadamard rotation of the embedding table (latent token embedding);
        // without this the draft head consumes embeddings in the rotated basis
        if (auto rot = lctx.model.hadamard_rotation(mtp_tok_embd); rot != nullptr) {
            cur = llm_build_hadamard_rotate(ctx0, cur, *rot, true);
        }
    } else {
        lctx.inp_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, batch.n_tokens);
        ggml_set_input(lctx.inp_embd);
        cur = lctx.inp_embd;
    }

    cb(cur, "inp_embd", -1);

    return cur;
}

struct ggml_tensor * llm_build_context::build_inp_mtp_states(int64_t n_hidden) {
    struct ggml_tensor * hidden_state = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_hidden, n_tokens);
    cb(hidden_state, "inp_mtp_states", -1);
    ggml_set_input(hidden_state);
    lctx.inp_mtp_states = hidden_state;
    return hidden_state;
}

struct ggml_tensor * llm_build_context::build_mtp_input(
        const struct llama_layer & mtp_layer,
        struct ggml_tensor * hidden_state,
        struct ggml_tensor * token_embd,
        int il,
    const char * output_name) {
    GGML_ASSERT(hidden_state->ne[0] == n_embd);
    GGML_ASSERT(ggml_are_same_shape(hidden_state, token_embd));

    struct ggml_tensor * hidden_state_norm = llm_build_norm(ctx0, hidden_state, hparams,
            mtp_layer.nextn.hnorm, nullptr, LLM_NORM_RMS, cb, il);
    struct ggml_tensor * token_emb_norm = llm_build_norm(ctx0, token_embd, hparams,
            mtp_layer.nextn.enorm, nullptr, LLM_NORM_RMS, cb, il);
    struct ggml_tensor * result;
    if (mtp_layer.nextn.eh_proj != nullptr) {
        struct ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
        cb(combined, "mtp_concat", il);
        result = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj, combined);
    } else {
        result = ggml_add(ctx0, token_emb_norm, hidden_state_norm);
    }
    if (output_name != nullptr) {
        cb(result, output_name, il);
    }
    return result;
}

ggml_tensor * llm_build_context::build_inp_pos() {
    int n_pos_per_embd = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
    lctx.inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, int64_t(n_tokens)*n_pos_per_embd);
    cb(lctx.inp_pos, "inp_pos", -1);
    ggml_set_input(lctx.inp_pos);
    return lctx.inp_pos;
}

ggml_tensor * llm_build_context::build_input_scale(int n_tokens) {
    int n_pos_per_token = 1;
    lctx.inp_scale = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens*n_pos_per_token);
    cb(lctx.inp_scale, "inp_scale", -1);
    ggml_set_input(lctx.inp_scale);
    return lctx.inp_scale;
}

ggml_tensor * llm_build_context::build_rope_factors(int il) {
    // choose long/short freq factors based on the context size
    const auto n_ctx_pre_seq = cparams.n_ctx / cparams.n_seq_max;

    if (model.layers[il].rope_freqs != nullptr) {
        return model.layers[il].rope_freqs;
    }

    if (n_ctx_pre_seq > hparams.n_ctx_orig_yarn) {
        return model.layers[il].rope_long;
    }

    return model.layers[il].rope_short;
}

ggml_tensor * llm_build_context::build_inp_out_ids() {
    lctx.inp_out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    cb(lctx.inp_out_ids, "inp_out_ids", -1);
    ggml_set_input(lctx.inp_out_ids);
    return lctx.inp_out_ids;
}

ggml_tensor * llm_build_context::build_inp_KQ_mask(bool causal) {
    if (causal && flash_attn) {
        lctx.inp_KQ_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        cb(lctx.inp_KQ_mask, "KQ_mask", -1);
        ggml_set_input(lctx.inp_KQ_mask);
        return lctx.inp_KQ_mask;
    }
    lctx.inp_KQ_mask = causal
        ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
        : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    cb(lctx.inp_KQ_mask, "KQ_mask", -1);
    ggml_set_input(lctx.inp_KQ_mask);

    return flash_attn ? ggml_cast(ctx0, lctx.inp_KQ_mask, GGML_TYPE_F16) : lctx.inp_KQ_mask;
}

ggml_tensor * llm_build_context::build_inp_KQ_mask_swa(bool causal) {
    GGML_ASSERT(hparams.n_swa > 0);
    if (causal && flash_attn) {
        lctx.inp_KQ_mask_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        cb(lctx.inp_KQ_mask_swa, "KQ_mask_swa", -1);
        ggml_set_input(lctx.inp_KQ_mask_swa);
        return lctx.inp_KQ_mask_swa;
    }

    lctx.inp_KQ_mask_swa = causal
        ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
        : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    cb(lctx.inp_KQ_mask_swa, "KQ_mask_swa", -1);
    ggml_set_input(lctx.inp_KQ_mask_swa);

    return flash_attn ? ggml_cast(ctx0, lctx.inp_KQ_mask_swa, GGML_TYPE_F16) : lctx.inp_KQ_mask_swa;
}

ggml_tensor * llm_build_context::build_inp_KQ_mask_swa_win(int64_t n_kv_win, bool causal) {
    GGML_ASSERT(hparams.n_swa > 0);
    GGML_ASSERT(n_kv_win > 0);
    if (causal && flash_attn) {
        lctx.inp_KQ_mask_swa_win = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv_win, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        cb(lctx.inp_KQ_mask_swa_win, "KQ_mask_swa_win", -1);
        ggml_set_input(lctx.inp_KQ_mask_swa_win);
        return lctx.inp_KQ_mask_swa_win;
    }

    lctx.inp_KQ_mask_swa_win = causal
        ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv_win,  GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
        : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    cb(lctx.inp_KQ_mask_swa_win, "KQ_mask_swa_win", -1);
    ggml_set_input(lctx.inp_KQ_mask_swa_win);

    return flash_attn ? ggml_cast(ctx0, lctx.inp_KQ_mask_swa_win, GGML_TYPE_F16) : lctx.inp_KQ_mask_swa_win;
}

ggml_tensor * llm_build_context::build_swa_mask_for_graph(uint32_t window, bool compacted, bool * windowed,
        const llama_kv_cache_fork * kv) {
    if (windowed) *windowed = false;
    lctx.swa_window_view = {};

    if (window == 0) {
        return nullptr;
    }

    const llama_kv_cache_fork & cache = kv ? *kv : kv_self;
    const int32_t cache_n_kv = kv ? (int32_t) kv->n : n_kv;

    const uint32_t pad = llama_kv_cache_fork::get_padding(cparams.flash_attn);
    const int64_t live = !compacted ? 0 : kv ? (int64_t) kv->live_swa()
        : (int64_t) swa_head - (int64_t) kv_self.sink_rows + n_tokens;
    const llama_swa_window_view view = compacted
        ? llama_swa_calc_window_view_compact(live, cache.sink_rows, n_tokens, window, pad)
        : llama_swa_calc_window_view(cache_n_kv, n_tokens, window, pad);

    if (!view.engaged) {
        return build_inp_KQ_mask_swa();
    }

    lctx.swa_window_view = {
        true,
        compacted,
        cache_n_kv,
        n_tokens,
        window,
        pad,
        view.w_view,
        view.win_off,
    };
    if (windowed) *windowed = true;
    return build_inp_KQ_mask_swa_win(view.w_view);
}

//build_mhc_post: x = 4096 x 4096 x 1 x 1, post = 4 x 4096 x 1 x 1, residual = 4096 x 4 x 4096 x 1, comb = 4 x 4 x 4096 x 1
//build_mhc_post: x = 4096 x 1 x 1 x 1,    post = 4 x 1 x 1 x 1,    residual = 4096 x 4 x 1 x 1,    comb = 4 x 4 x 1 x 1
// x        = n_embd x n_tokens           <--- y in Pangu
// post     = 4 x n_tokens                <--- h_post in Pangu
// residual = n_embd x 4 x n_tokens       <--- Rin in Pangu
// comb     = 4 x 4 x n_tokens

ggml_tensor * llm_build_context::build_mhc_post(
        ggml_tensor * x,
        ggml_tensor * post,
        ggml_tensor * residual,
        ggml_tensor * comb,
        int64_t n_embd,
        int64_t n_stream,
        bool comb_output_dim0) {
    const int64_t n_tokens = x->ne[1];
    ggml_tensor * out = nullptr;

    //ggml_tensor * x3 = ggml_reshape_3d(ctx0, x, n_embd, 1, n_tokens);
    //ggml_tensor * post3 = ggml_reshape_3d(ctx0, post, 1, n_stream, n_tokens);
    //ggml_tensor repeater;
    //repeater.ne[0] = n_embd; repeater.ne[1] = n_stream; repeater.ne[2] = n_tokens; repeater.ne[3] = 1;
    //ggml_tensor * term1 = ggml_mul(ctx0, ggml_repeat(ctx0, x3, &repeater), post3);

    //ggml_tensor * term2 = nullptr;
    //for (int s = 0; s < n_stream; ++s) {
    //    ggml_tensor * m_s = ggml_cont(ctx0, ggml_view_2d(ctx0, comb, n_stream, n_tokens, comb->nb[2], s*comb->nb[1]));
    //    ggml_tensor * m_s3 = ggml_reshape_3d(ctx0, m_s, 1, n_stream, n_tokens);
    //    ggml_tensor * acc = ggml_mul(ctx0, residual, m_s3);
    //    ggml_tensor * summed = ggml_sum_rows_ext(ctx0, acc, 1);
    //    term2 = term2 ? ggml_concat(ctx0, term2, summed, 1) : summed;
    //}
    //return ggml_add(ctx0, term1, term2);
    //printf("%s: x = %ld x %ld x %ld x %ld, post = %ld x %ld x %ld x %ld, residual = %ld x %ld x %ld x %ld, comb = %ld x %ld x %ld x %ld\n",
    //        __func__, x->ne[0], x->ne[1], x->ne[2], x->ne[3], post->ne[0], post->ne[1], post->ne[2], post->ne[3],
    //        residual->ne[0], residual->ne[1], residual->ne[2], residual->ne[3],
    //        comb->ne[0], comb->ne[1], comb->ne[2], comb->ne[3]);

    return ggml_hc_post(ctx0, x, post, residual, comb);

    for (int64_t dst = 0; dst < n_stream; ++dst) {
        ggml_tensor * post_dst = ggml_cont(ctx0,
                ggml_view_2d(ctx0, post, 1, n_tokens, post->nb[1], dst*post->nb[0]));
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < n_stream; ++src) {
            ggml_tensor * res_src = ggml_cont(ctx0,
                    ggml_view_2d(ctx0, residual, n_embd, n_tokens,
                            residual->nb[2], src*residual->nb[1]));
            const size_t comb_offset = comb_output_dim0
                    ? dst*comb->nb[0] + src*comb->nb[1]
                    : src*comb->nb[0] + dst*comb->nb[1];
            ggml_tensor * comb_dst_src = ggml_cont(ctx0,
                    ggml_view_2d(ctx0, comb, 1, n_tokens, comb->nb[2], comb_offset));
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_dst_src));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        if (out) {
            out = ggml_concat(ctx0, out, cur, 1);
            cb(out, "mhc_post", dst);
        } else {
            out = cur;
        }
        //out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llm_build_context::build_mhc_weighted_sum(
        ggml_tensor * x,
        ggml_tensor * weights,
        int64_t n_embd,
        int64_t n_stream) {
    const int64_t n_tokens = x->ne[2];
    ggml_tensor * out = nullptr;

    if (weights->ne[3] == 1 && x->ne[3] == 1) {
        auto w = ggml_reshape_4d(ctx0, weights, 1, weights->ne[0], weights->ne[1], weights->ne[2]);
        return ggml_mul_multi_add(ctx0, x, w);
    }
    for (int64_t stream = 0; stream < n_stream; ++stream) {
        ggml_tensor * x_stream = ggml_cont(ctx0,
                ggml_view_2d(ctx0, x, n_embd, n_tokens,
                        x->nb[2], stream*x->nb[1]));
        ggml_tensor * weight = ggml_cont(ctx0,
                ggml_view_2d(ctx0, weights, 1, n_tokens,
                        weights->nb[1], stream*weights->nb[0]));
        ggml_tensor * cur = ggml_mul(ctx0, x_stream, weight);
        out = out ? ggml_add(ctx0, out, cur) : cur;
    }

    return out;
}

ggml_tensor * llm_build_context::build_mhc_pre_projection(
        ggml_tensor * x,
        ggml_tensor * fn,
        ggml_tensor * gamma,
        int64_t n_embd,
        int64_t n_stream,
        float norm_rms_eps,
        bool force_contiguous) {
    if (force_contiguous && !ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, n_embd*n_stream, x->ne[2]);
    ggml_tensor * normed = gamma ? ggml_fused_rms_norm(ctx0, flat, gamma, hparams.f_norm_rms_eps) :
                                   ggml_rms_norm(ctx0, flat, norm_rms_eps);

    return ggml_mul_mat(ctx0, fn, normed);
}

ggml_tensor * llm_build_context::build_inp_mean() {
    lctx.inp_mean = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
    cb(lctx.inp_mean, "inp_mean", -1);
    ggml_set_input(lctx.inp_mean);
    return lctx.inp_mean;
}

ggml_tensor * llm_build_context::build_inp_cls() {
    lctx.inp_cls = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    cb(lctx.inp_cls, "inp_cls", -1);
    ggml_set_input(lctx.inp_cls);
    return lctx.inp_cls;
}

ggml_tensor * llm_build_context::build_inp_s_copy() {
    lctx.inp_s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, kv_self.size);
    cb(lctx.inp_s_copy, "inp_s_copy", -1);
    ggml_set_input(lctx.inp_s_copy);
    return lctx.inp_s_copy;
}

ggml_tensor * llm_build_context::build_inp_s_mask() {
    lctx.inp_s_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_kv);
    cb(lctx.inp_s_mask, "inp_s_mask", -1);
    ggml_set_input(lctx.inp_s_mask);
    return lctx.inp_s_mask;
}

ggml_tensor * llm_build_context::build_inp_s_seq() {
    lctx.inp_s_seq = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    cb(lctx.inp_s_seq, "inp_s_seq", -1);
    ggml_set_input(lctx.inp_s_seq);
    return lctx.inp_s_seq;
}

ggml_cgraph * llm_build_context::append_pooling(struct ggml_cgraph * gf) {
    // find result_norm tensor for input
    struct ggml_tensor * inp = nullptr;
    if (lctx.cparams.mtp) {
        for (int i = gf->n_nodes - 1; i >= 0; --i) {
            if (strcmp(gf->nodes[i]->name, "result_mtp_embd") == 0) {
                inp = gf->nodes[i];
                break;
            }
        }
    }
    if (!inp) {
        for (int i = gf->n_nodes - 1; i >= 0; --i) {
            inp = gf->nodes[i];

            if (strcmp(inp->name, "result_norm") == 0 ||
                strcmp(inp->name, "result_embd") == 0 ||
                strcmp(inp->name, "output_normed") == 0) {
                break;
            }
            inp = nullptr;
        }
    }
    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");
    const bool is_mtp_hidden = strcmp(inp->name, "result_mtp_embd") == 0;

    struct ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_MEAN:
            {
                struct ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_ATTENTION:
            {
                // NV-Embed-style param-free latent-attention pooling, in graph. Port of orphan 9ea7e37cc/32b9d3344
                // (own code), with the cross-sequence leak of that version fixed: the orphan softmaxed over ALL
                // tokens in the ubatch, so with several sequences a sequence's weights leaked onto the others'
                // tokens. Here a per-sequence additive mask (0 for the sequence's own tokens, -1e9 elsewhere,
                // derived from inp_mean's non-zero pattern) makes the pool exact per sequence.
                //   q[:,s]      = mean of the sequence's token states (inp_mean)
                //   scores[t,s] = inp[:,t] . q[:,s]
                //   out[:,s]    = sum_t softmax_t(scores[t,s]) * inp[:,t]
                struct ggml_tensor * inp_mean = build_inp_mean();
                struct ggml_tensor * inpT = ggml_cont(ctx0, ggml_transpose(ctx0, inp));   // [n_tokens, n_embd]
                struct ggml_tensor * q = ggml_mul_mat(ctx0, inpT, inp_mean);              // [n_embd, n_seq]
                struct ggml_tensor * scores = ggml_mul_mat(ctx0, inp, q);                 // [n_tokens, n_seq]
                struct ggml_tensor * seq_mask = ggml_scale_bias(ctx0, ggml_step(ctx0, inp_mean), 1e9f, -1e9f);
                scores = ggml_soft_max_ext(ctx0, ggml_add(ctx0, scores, seq_mask), nullptr, 1.0f, 0.0f);
                cur = ggml_mul_mat(ctx0, inpT, scores);                                   // [n_embd, n_seq]
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                struct ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                // ported from upstream llm_graph_context::build_pooling (also in llama-graph.cpp, which the
                // live path does not use); llama-context already extracts n_cls_out floats per sequence
                if (model.arch == LLM_ARCH_MODERNBERT) {
                    // gte reranker: mean first, then the prediction head
                    struct ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    struct ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }
                // classification head
                if (model.cls) {
                    cur = ggml_mul_mat(ctx0, model.cls, cur);
                    if (model.cls_b) {
                        cur = ggml_add(ctx0, cur, model.cls_b);
                    }
                    cur = model.arch == LLM_ARCH_MODERNBERT ? ggml_gelu(ctx0, cur) : ggml_tanh(ctx0, cur);
                    if (model.cls_norm) {
                        cur = llm_build_norm(ctx0, cur, hparams, model.cls_norm, NULL, LLM_NORM, cb, -1);
                    }
                }
                // single-layer heads have no cls.output, e.g. jina-reranker-v1-tiny-en
                if (model.cls_out) {
                    cur = ggml_mul_mat(ctx0, model.cls_out, cur);
                    if (model.cls_out_b) {
                        cur = ggml_add(ctx0, cur, model.cls_out_b);
                    }
                }
                if (model.arch == LLM_ARCH_QWEN3 || model.arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur); // qwen3 reranker
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    // SentenceTransformers dense projection (dense_2 -> dense_3), applied AFTER pooling.
    // SOURCE-READ: src/llama-graph.cpp:3630-3648, llm_graph_context::build_dense_out (the
    // new-gen path's version of this, called from llama_model_base::build_graph() -- unreachable
    // end to end, same gap as the rest of S69-OLLAMA-FORMAT-EMBEDDINGGEMMA-316-VS-314-TENSORS-
    // 2026-09-28-01). Ported verbatim here rather than called, for the same reason: nothing
    // constructs a llama_model_base for any arch (memory/project-fork-models-wired-but-no-
    // dispatcher.md). model.dense_2_out_layers/dense_3_out_layers are TENSOR_NOT_REQUIRED
    // (create_gemma_embedding_tensors, llama-load-tensors.cpp), so this is a no-op for every
    // arch except one converted with --sentence-transformers-dense-modules.
    if (model.dense_2_out_layers || model.dense_2_out_layers_b || model.dense_3_out_layers) {
        if (model.dense_2_out_layers) {
            cur = ggml_mul_mat(ctx0, model.dense_2_out_layers, cur);
        }
        if (model.dense_2_out_layers_b) {
            cur = ggml_add(ctx0, cur, model.dense_2_out_layers_b);
        }
        if (model.dense_3_out_layers) {
            cur = ggml_mul_mat(ctx0, model.dense_3_out_layers, cur);
        }
        cb(cur, "result_embd_pooled", -1);
    } else if (!is_mtp_hidden || pooling_type != LLAMA_POOLING_TYPE_NONE) {
        cb(cur, "result_embd_pooled", -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_tensor * llm_build_context::llm_build_pos_bucket(bool causal) {
    if (causal) {
        lctx.inp_pos_bucket = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv,     n_tokens);
    } else {
        lctx.inp_pos_bucket = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    }

    ggml_set_input(lctx.inp_pos_bucket);
    cb(lctx.inp_pos_bucket, "pos_bucket", -1);

    return lctx.inp_pos_bucket;
}

ggml_tensor * llm_build_context::llm_build_pos_bias(struct ggml_tensor * pos_bucket, struct ggml_tensor * attn_rel_b) {
    struct ggml_tensor * pos_bucket_1d = ggml_view_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1], 0);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    struct ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);
    cb(pos_bias, "pos_bias", -1);

    pos_bias = ggml_view_3d(ctx0, pos_bias, pos_bias->ne[0], lctx.inp_pos_bucket->ne[0], lctx.inp_pos_bucket->ne[1], ggml_element_size(pos_bias) * pos_bias->ne[0], ggml_element_size(pos_bias) * pos_bias->ne[0] * lctx.inp_pos_bucket->ne[0],  0);
    cb(pos_bias, "pos_bias", -1);

    pos_bias = ggml_permute(ctx0, pos_bias, 2, 0, 1, 3);
    cb(pos_bias, "pos_bias", -1);

    pos_bias = ggml_cont(ctx0, pos_bias);
    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

// Prism ternary Hadamard transform (forward: H(Dx), inverse: D(Hz))
static struct ggml_tensor * llm_build_hadamard_rotate(
        struct ggml_context * ctx0,
        struct ggml_tensor * cur,
        const llama_hadamard_transform & t,
        bool inverse = false) {
    if (t.block_size == 0) {
        return cur;
    }
    struct ggml_tensor * res = cur;
    if (res->type != GGML_TYPE_F32) {
        res = ggml_cast(ctx0, res, GGML_TYPE_F32);
    }
    if (!inverse && t.perm_rep > 1) {
        // tiled [hd, nk, rep] -> grouped [hd, rep, nk] feature order (GDN v-grouped)
        struct ggml_tensor * x = ggml_is_contiguous(res) ? res : ggml_cont(ctx0, res);
        const int64_t ne1 = x->ne[1], ne2 = x->ne[2], ne3 = x->ne[3];
        x = ggml_reshape_4d(ctx0, x, t.perm_hd, t.perm_nk, t.perm_rep, ne1*ne2*ne3);
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
        res = ggml_reshape_4d(ctx0, x, t.perm_hd*t.perm_nk*t.perm_rep, ne1, ne2, ne3);
    }
    const bool signs_first = !inverse;
    if (signs_first && t.signs) {
        res = ggml_mul(ctx0, res, t.signs);
    }
    const int64_t n = t.block_size;
    GGML_ASSERT(n > 0 && ggml_nelements(res) % n == 0);
    struct ggml_tensor * rot;
    if (!ggml_is_contiguous(res)) {
        rot = ggml_cont_2d(ctx0, res, n, ggml_nelements(res)/n);
    } else {
        rot = ggml_reshape_2d(ctx0, res, n, ggml_nelements(res)/n);
    }
    rot = ggml_hadamard(ctx0, rot, (int) n);
    struct ggml_tensor * out = ggml_reshape_4d(ctx0, rot, res->ne[0], res->ne[1], res->ne[2], res->ne[3]);
    if (!signs_first && t.signs) {
        out = ggml_mul(ctx0, out, t.signs);
    }
    return out;
}

// Two Prism transforms are interchangeable when every field matches. llama_hadamard_transform has no operator== in the fork
// (upstream's does); the signs tensors are shared per width by the loader (llama-model-loader.cpp make_signs), so comparing the
// pointer is the comparison of the data.
static bool llm_hadamard_same(const llama_hadamard_transform & a, const llama_hadamard_transform & b) {
    return a.signs == b.signs && a.block_size == b.block_size &&
           a.perm_hd == b.perm_hd && a.perm_nk == b.perm_nk && a.perm_rep == b.perm_rep;
}

static ggml_tensor * llm_lora_seq_mask(llama_context & lctx, ggml_context * ctx0, llama_adapter_lora * ad, int64_t n_rows);
static ggml_tensor * llm_lora_gate_mask(llama_context & lctx, ggml_context * ctx0, llama_adapter_lora * ad, int64_t n_rows);

ggml_tensor * llm_build_context::llm_build_inp_embd(
        struct ggml_context * ctx,
       struct llama_context & lctx,
        const llama_hparams & hparams,
          const llama_batch & batch,
         struct ggml_tensor * tok_embd,
         const llm_build_cb & cb) {
    const int64_t n_embd = hparams.n_embd;

    struct ggml_tensor * inpL;

    lctx.inp_engram_gate_mask = nullptr;
    lctx.inp_engram_rows.clear();
    lctx.inp_engram_gate_ids.clear();
    if (batch.token) {
        lctx.inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);

        inpL = ggml_get_rows(ctx, tok_embd, lctx.inp_tokens);

        // undo the Hadamard rotation of the embedding table
        if (auto rot = lctx.model.hadamard_rotation(tok_embd); rot != nullptr) {
            inpL = llm_build_hadamard_rotate(ctx, inpL, *rot, true);
        }

        // Embedding LoRA (token_embd, A/B in lookup layout): delta = s * B (A[:, token]) added to the looked-up row BEFORE the
        // Granite / Gemma embedding scale below, as PEFT scales the delta with the row (lora/layer.py Embedding.forward).
        // The same adapter weights are skipped by llm_build_lora_add so a tied lm_head never takes them as a linear LoRA.
        auto add_embd = [&](llama_adapter_lora * ad, float alpha_scale, ggml_tensor * mask_in) {
            llama_adapter_lora_weight * lw = ad->get_weight(tok_embd);
            if (lw == nullptr || !lw->is_embd) {
                return;
            }
            ggml_tensor * d = ggml_mul_mat(ctx, lw->b, ggml_get_rows(ctx, lw->a, lctx.inp_tokens));
            d = ggml_scale(ctx, d, lw->get_scale(ad->alpha, alpha_scale));
            if (mask_in != nullptr) {
                d = ggml_mul(ctx, d, mask_in);
            }
            if (ggml_tensor * gm = llm_lora_gate_mask(lctx, ctx, ad, batch.n_tokens)) {
                d = ggml_mul(ctx, d, gm);
            }
            inpL = ggml_add(ctx, inpL, d);
        };
        if (lctx.loras) {
            for (const auto & it : *lctx.loras) {
                add_embd(it.first, it.second, nullptr);
            }
        }
        for (llama_adapter_lora * ad : lctx.seq_loras.adapters) {
            add_embd(ad, 1.0f, llm_lora_seq_mask(lctx, ctx, ad, batch.n_tokens));
        }
    } else {
       lctx.inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, batch.n_tokens);
        inpL = lctx.inp_embd;
        ggml_set_input(lctx.inp_embd);
    }

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx, inpL, hparams.f_embedding_scale);
    }

    cb(inpL, "inp_embd", -1);

    return inpL;
}

void llm_build_context::llm_build_kv_store(
       struct llama_context & lctx,
        struct ggml_context * ctx,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
       const llama_kv_cache_fork & kv,
         struct ggml_cgraph * graph,
         struct ggml_tensor * k_cur,
         struct ggml_tensor * v_cur,
                    int32_t   n_tokens,
                    int32_t   kv_head,
         const llm_build_cb & cb,
                    int64_t   il) {
    const int64_t n_ctx = cparams.n_ctx;
    const int32_t n_cache_rows = kv.rows(il);

    //const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    const int64_t n_head_kv     = hparams.n_head_kv(il);
    const int64_t n_embd_head_k = hparams.n_embd_head_k(il);

    GGML_ASSERT(kv.size == n_ctx);

    //struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv.k_l[il], n_tokens*n_embd_k_gqa,
    //        (ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa))*kv_head);
    //cb(k_cache_view, "k_cache_view", il);

    if (k_cur) {
        GGML_ASSERT(2*il+1 < (int)lctx.cache_copies.size());
        auto k_row_size = ggml_row_size(kv.k_l[il]->type, n_embd_head_k);
        ggml_tensor * k_cache_view = ggml_view_2d(ctx, kv.k_l[il], n_embd_head_k, n_tokens*n_head_kv,
                k_row_size, k_row_size*n_head_kv*kv_head);

        lctx.cache_copies[2*il+0].cpy  = ggml_cpy(ctx, k_cur, k_cache_view);
        lctx.cache_copies[2*il+0].step = k_row_size*n_head_kv;

        // note: storing RoPE-ed version of K in the KV cache
        ggml_build_forward_expand(graph, lctx.cache_copies[2*il+0].cpy);

        // FFD (--ffd): re-quantise every ffd_block-cell block this ubatch wrote into the layer's 2-bit K shadow. The
        // written cells are [kv_head, kv_head + n_tokens) -- this store is a contiguous copy, not a scatter -- and reach
        // the op through inp_ffd_idx, filled per ubatch from kv_self.head in llama_set_inputs, so a reused graph (whose
        // cache_copies offsets are re-patched) still quantises the right blocks. src[4] = the K store: the op ignores
        // it, but the edge orders the re-quantisation after the write it must read.
        if (il < (int64_t) kv.ffd_k2.size() && kv.ffd_k2[il]) {
            if (!lctx.inp_ffd_idx) {
                lctx.inp_ffd_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
                ggml_set_name(lctx.inp_ffd_idx, "inp_ffd_idx");
                ggml_set_input(lctx.inp_ffd_idx);
            }
            ggml_tensor * k_rows = ggml_reshape_2d(ctx, kv.k_l[il], n_embd_head_k*n_head_kv, kv.rows(il));
            ggml_tensor * k2_new = ggml_ffd_quant_k(ctx, k_rows, lctx.inp_ffd_idx, kv.ffd_k2[il], kv.ffd_ks[il],
                                                    (int) n_embd_head_k, kv.ffd_block);
            k2_new->src[4] = lctx.cache_copies[2*il+0].cpy;
            cb(k2_new, "ffd_quant_k", il);
            ggml_build_forward_expand(graph, k2_new);
        }
    }

    if (v_cur) {
        ggml_tensor * v_cache_view = nullptr;
        if (!kv.v_trans) {
            v_cache_view = ggml_view_1d(ctx, kv.v_l[il], n_tokens*n_embd_v_gqa,
                    (kv_head)*ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa));
            lctx.cache_copies[2*il+1].step = ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa);
        } else {
            // note: the V cache is transposed for legacy non-FA layouts
            v_cache_view = ggml_view_2d(ctx, kv.v_l[il], n_tokens, n_embd_v_gqa,
                    (n_cache_rows)*ggml_element_size(kv.v_l[il]),
                    (kv_head)*ggml_element_size(kv.v_l[il]));
            lctx.cache_copies[2*il+1].step = ggml_element_size(kv.v_l[il]);

            // FLATTEN BEFORE TRANSPOSING, as upstream does. Builders that RMS-norm V per head (gemma3/4,
            // gemma3n, ...) hand over a 3-D [n_embd_head_v, n_head_kv, n_tokens] v_cur; ggml_transpose swaps
            // only dims 0 and 1, giving [n_head_kv, n_embd_head_v, n_tokens], and ggml_cpy then fills the
            // [n_tokens, n_embd_v_gqa] view in that tensor's order -- every element present, in the wrong
            // cell. MEASURED 2026-09-28: gemma-4-E2B greedy was garbage with -fa off on CPU and CUDA and
            // ' Paris' with -fa on (the untransposed store is a 1-D view and never hit this); eval-callback
            // showed q/k/v sums identical in both modes and the first divergence at kq/softmax/kqv.
            // ggml_n_dims() counts only dims with more than one element, so a ONE-token ubatch ([d, n_head_kv, 1]) reported 2
            // and skipped the flatten: single-token decode with -fa off stored every V head interleaved (found 2026-10-05 by
            // test-llama-archs --parity, gemma4: step-vs-batch NMSE 2.0, interleaved kqv rows). Test the shape instead.
            if (v_cur->ne[0] != n_embd_v_gqa || v_cur->ne[2] != 1 || v_cur->ne[3] != 1) {
                if (!ggml_is_contiguous(v_cur)) {
                    v_cur = ggml_cont(ctx, v_cur);
                }
                v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_v_gqa, n_tokens);
            }
            v_cur = ggml_transpose(ctx, v_cur);
        }
        cb(v_cache_view, "v_cache_view", il);

        lctx.cache_copies[2*il+1].cpy  = ggml_cpy(ctx, v_cur, v_cache_view);
        ggml_build_forward_expand(graph, lctx.cache_copies[2*il+1].cpy);
    }
}

// The Lily same-sequence averaging matrix (row S69-ADAPTERS-W2-LILY-PER-SEQUENCE-ROUTER-2026-10-26-01), one per graph. The router mean
// needs every token of the sequence, so rows pruned to the outputs (inp_out_ids) cannot be served: llama_decode makes every token an
// output while a Lily adapter is active, which keeps n_rows == n_tokens here.
static ggml_tensor * llm_lily_seq_mean(llama_context & lctx, ggml_context * ctx0, int64_t n_rows) {
    const int64_t n_tokens = lctx.lora_seq_n_tokens;
    if (n_rows != n_tokens) {
        GGML_ABORT("Lily adapter on an activation of %lld rows in a graph of %lld tokens: rows were pruned to the outputs, so the "
                   "per-sequence router mean is not computable (llama_decode should have made every token an output)",
                   (long long) n_rows, (long long) n_tokens);
    }
    if (lctx.inp_lily_seq_mean == nullptr) {
        lctx.inp_lily_seq_mean = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
        ggml_set_name(lctx.inp_lily_seq_mean, "lily_seq_mean");
        ggml_set_input(lctx.inp_lily_seq_mean);
    }
    return lctx.inp_lily_seq_mean;
}

// The per-token scale mask of `ad` for an activation with `n_rows` rows in the graph being built: one F32 [1, n_tokens] input per adapter of the
// per-sequence union. After the last layer's rows were pruned to the outputs (ggml_get_rows with inp_out_ids) the mask is pruned the same way.
static ggml_tensor * llm_lora_seq_mask(llama_context & lctx, ggml_context * ctx0, llama_adapter_lora * ad, int64_t n_rows) {
    const int64_t n_tokens = lctx.lora_seq_n_tokens;
    ggml_tensor * full = nullptr;
    for (auto & m : lctx.lora_seq_masks) {
        if (m.first == ad) {
            full = m.second;
            break;
        }
    }
    if (full == nullptr) {
        full = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        ggml_set_name(full, "lora_seq_mask");
        ggml_set_input(full);
        lctx.lora_seq_masks.emplace_back(ad, full);
    }
    if (n_rows == n_tokens) {
        return full;
    }
    GGML_ASSERT(lctx.inp_out_ids != nullptr && lctx.inp_out_ids->ne[0] == n_rows &&
                "per-sequence LoRA: an activation has neither n_tokens rows nor the pruned n_outputs rows");
    return ggml_get_rows(ctx0, full, lctx.inp_out_ids); // [1, n_outputs]
}

// The per-token 0/1 gate of `ad` (adapter.lora.gate != none): one F32 [1, n_tokens] input filled by llama_set_inputs from the token ids and the
// llama_set_draft_start positions; nullptr for an ungated adapter. Pruned to the outputs like the per-sequence mask.
static ggml_tensor * llm_lora_gate_mask(llama_context & lctx, ggml_context * ctx0, llama_adapter_lora * ad, int64_t n_rows) {
    if (ad->gate == llama_adapter_lora::LLAMA_LORA_GATE_NONE) {
        return nullptr;
    }
    const int64_t n_tokens = lctx.lora_seq_n_tokens;
    ggml_tensor * full = nullptr;
    for (auto & m : lctx.lora_gate_masks) {
        if (m.first == ad) {
            full = m.second;
            break;
        }
    }
    if (full == nullptr) {
        full = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
        ggml_set_name(full, "lora_gate_mask");
        ggml_set_input(full);
        lctx.lora_gate_masks.emplace_back(ad, full);
    }
    if (n_rows == n_tokens) {
        return full;
    }
    GGML_ASSERT(lctx.inp_out_ids != nullptr && lctx.inp_out_ids->ne[0] == n_rows &&
                "gated LoRA: an activation has neither n_tokens rows nor the pruned n_outputs rows");
    return ggml_get_rows(ctx0, full, lctx.inp_out_ids);
}

ggml_tensor * llm_build_context::llm_build_lora_mm(
        struct llama_context & lctx,
         struct ggml_context * ctx0,
          struct ggml_tensor * w,
          struct ggml_tensor * cur) {
    struct ggml_tensor * cur_mm = cur;
    if (auto rot = lctx.model.hadamard_rotation(w); rot != nullptr) {
        // several folded weights consume the same input activation; rotate it once per graph.
        // Key by content (signs tensor + transform params), not by the per-weight transform pointer
        const auto memo_key = std::make_tuple((const struct ggml_tensor *) cur, (const struct ggml_tensor *) rot->signs,
                                              rot->block_size, rot->perm_hd, rot->perm_nk, rot->perm_rep);
        const auto memo_it  = lctx.hadamard_memo.find(memo_key);
        if (memo_it != lctx.hadamard_memo.end()) {
            cur_mm = memo_it->second;
        } else {
            cur_mm = llm_build_hadamard_rotate(ctx0, cur, *rot);
            lctx.hadamard_memo[memo_key] = cur_mm;
        }
    }
    // IA3 (row S69-ADAPTERS-LINEAR-WRAPPER-IA3-RED-VEC-SCALE-2026-10-26-01): the input-side scale multiplies the activation BEFORE the
    // matmul (PEFT feedforward modules: base(x * l), peft a7e8ea4a1227 tuners/ia3/layer.py:172-201)
    if (lctx.loras) {
        for (const auto & it : *lctx.loras) {
            llama_adapter_lora_weight * lw = it.first->get_weight(w);
            if (lw != nullptr && lw->ia3_in != nullptr) {
                cur_mm = ggml_mul(ctx0, cur_mm, llama_ia3_vec(ctx0, lw->ia3_in, it.second));
            }
        }
    }
    struct ggml_tensor * res = ggml_mul_mat(ctx0, w, cur_mm);
    // NVFP4 per-tensor scale (attached by the loader from "<name>.scale"): applied to the product, BEFORE any LoRA
    // residual and before the caller's bias, exactly where ModelOpt's scale_2 belongs.
    if (struct ggml_tensor * ws = lctx.model.weight_scale(w)) {
        res = ggml_mul(ctx0, res, ws);
    }
    // IA3 output-side scale: base(x) * l. The loader refuses a biased base layer (PEFT scales the bias too; callers add it after this).
    if (lctx.loras) {
        for (const auto & it : *lctx.loras) {
            llama_adapter_lora_weight * lw = it.first->get_weight(w);
            if (lw != nullptr && lw->ia3_out != nullptr) {
                res = ggml_mul(ctx0, res, llama_ia3_vec(ctx0, lw->ia3_out, it.second));
            }
        }
    }
    return llm_build_lora_add(lctx, ctx0, w, cur, res);
}

bool llm_build_context::llm_lora_targets(llama_context & lctx, ggml_tensor * w) {
    if (!w) {
        return false;
    }
    for (auto & it : lctx.lora_adapters) {
        if (it.first->get_weight(w)) {
            return true;
        }
    }
    if (lctx.loras) {
        for (const auto & it : *lctx.loras) {
            if (auto * lw = it.first->get_weight(w); lw && !lw->is_embd) {
                return true;
            }
        }
    }
    for (llama_adapter_lora * ad : lctx.seq_loras.adapters) {
        if (auto * lw = ad->get_weight(w); lw && !lw->is_embd) {
            return true;
        }
    }
    return false;
}

ggml_tensor * llm_build_context::llm_build_lora_add(
        struct llama_context & lctx,
         struct ggml_context * ctx0,
          struct ggml_tensor * w,
          struct ggml_tensor * cur,
          struct ggml_tensor * res) {
    for (auto & it : lctx.lora_adapters) {
        struct llama_lora_weight * lora = it.first->get_weight(w);
        if (lora == nullptr) {
            continue;
        }
        const float alpha = it.first->alpha;
        const float rank  = (float) lora->b->ne[0];
        const float scale = alpha ? it.second * alpha / rank : it.second;
        struct ggml_tensor * ab_cur = ggml_mul_mat(
            ctx0, lora->b,
            ggml_mul_mat(ctx0, lora->a, cur)
        );
        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }
    // Adapters of the llama_adapter_lora API (llama_set_adapters_lora -- what common_set_adapter_lora, --lora and llama-server use). The loop above
    // serves only the older llama_lora_adapter API; before this loop existed the server's adapters were silently ignored by this graph builder.
    if (lctx.loras) {
        for (const auto & it : *lctx.loras) {
            llama_adapter_lora_weight * lw = it.first->get_weight(w);
            if (lw == nullptr || lw->is_embd) {  // an embedding-layout pair is not a linear LoRA (tied lm_head shares the name)
                continue;
            }
            if (lw->is_ia3()) {  // IA3 scales are applied around the matmul in llm_build_lora_mm, not added here
                continue;
            }
            if (lw->lily_router) {
                // Lily: per-sequence router mean; PEFT's `scaling` is folded into B, the adapter scale c multiplies the delta
                GGML_ASSERT(cur->ne[2] == 1 && cur->ne[3] == 1 && "Lily needs a [n_in, n_tokens] activation");
                struct ggml_tensor * ly = llama_lily_delta(ctx0, *lw, cur, llm_lily_seq_mean(lctx, ctx0, cur->ne[1]));
                if (it.second != 1.0f) {
                    ly = ggml_scale(ctx0, ly, it.second);
                }
                if (ggml_tensor * gm = llm_lora_gate_mask(lctx, ctx0, it.first, cur->ne[1])) {
                    ly = ggml_mul(ctx0, ly, gm);
                }
                res = ggml_add(ctx0, res, ly);
                continue;
            }
            struct ggml_tensor * ab_cur = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur));
            if (lw->dora_dm1) {
                // DoRA: delta = (res + s BAx) * (d - 1) + s BAx, then the adapter scale c on the whole delta
                ab_cur = llama_dora_delta(ctx0, *lw, res, ggml_scale(ctx0, ab_cur, lw->get_scale(it.first->alpha, 1.0f)));
                if (it.second != 1.0f) {
                    ab_cur = ggml_scale(ctx0, ab_cur, it.second);
                }
            } else {
                ab_cur = ggml_scale(ctx0, ab_cur, lw->get_scale(it.first->alpha, it.second));
            }
            if (ggml_tensor * gm = llm_lora_gate_mask(lctx, ctx0, it.first, cur->ne[1])) {
                GGML_ASSERT(cur->ne[2] == 1 && cur->ne[3] == 1 && "gated LoRA needs a [n_in, n_tokens] activation");
                ab_cur = ggml_mul(ctx0, ab_cur, gm); // [n_out, n_tokens] * [1, n_tokens]
            }
            res = ggml_add(ctx0, res, ab_cur);
        }
    }
    // Per-sequence adapters (llama_set_seq_adapters_lora): every adapter of the union adds its branch, multiplied per token by the scale its token's
    // sequence gave it (0 = the sequence does not use it)
    for (llama_adapter_lora * ad : lctx.seq_loras.adapters) {
        llama_adapter_lora_weight * lw = ad->get_weight(w);
        if (lw == nullptr || lw->is_embd) {
            continue;
        }
        GGML_ASSERT(!lw->is_ia3() && "IA3 adapters are context-wide only (llama_set_adapters_lora); a per-sequence IA3 scale is not supported");
        GGML_ASSERT(cur->ne[2] == 1 && cur->ne[3] == 1 && "per-sequence LoRA needs a [n_in, n_tokens] activation");
        struct ggml_tensor * mask   = llm_lora_seq_mask(lctx, ctx0, ad, cur->ne[1]);
        if (lw->lily_router) {
            // Lily per sequence: the router mean is already per sequence; the mask keeps the delta to the sequences using `ad`
            struct ggml_tensor * ly = llama_lily_delta(ctx0, *lw, cur, llm_lily_seq_mean(lctx, ctx0, cur->ne[1]));
            res = ggml_add(ctx0, res, ggml_mul(ctx0, ly, mask));
            continue;
        }
        struct ggml_tensor * ab_cur = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur));
        ab_cur = ggml_scale(ctx0, ab_cur, lw->get_scale(ad->alpha, 1.0f)); // alpha / rank of this tensor
        if (lw->dora_dm1) {
            ab_cur = llama_dora_delta(ctx0, *lw, res, ab_cur);              // DoRA delta; the mask (per-seq scale) applies to all of it
        }
        ab_cur = ggml_mul(ctx0, ab_cur, mask);                              // [n_out, n_tokens] * [1, n_tokens]
        if (ggml_tensor * gm = llm_lora_gate_mask(lctx, ctx0, ad, cur->ne[1])) {
            ab_cur = ggml_mul(ctx0, ab_cur, gm);
        }
        res = ggml_add(ctx0, res, ab_cur);
    }
    return res;
}

ggml_tensor * llm_build_context::build_mla_output_gate(
        llama_context & lctx,
        ggml_context * ctx0,
        ggml_tensor * output,
        ggml_tensor * input,
        ggml_tensor * gate_w,
        int64_t head_dim,
        int64_t n_head,
        int il,
        const llm_build_cb & cb) {
    // Two gate granularities exist in the wild, told apart by the projection's own width
    // rather than by architecture, so neither needs to name the other:
    //   per-head      gate_w is [n_embd, n_head]            -> one scalar per head
    //                 (bailingmoe3: [1024, 8])
    //   per-head-dim  gate_w is [n_embd, n_head * head_dim] -> one scalar per head DIMENSION
    //                 (kimi-k3:    [1024, 512])
    // Reshaping a per-head-dim projection to [1, n_head, T] would silently reinterpret 512
    // values as 8, so the width decides and anything else asserts.
    const int64_t gate_w_out = gate_w->ne[1];
    const bool per_head_dim  = gate_w_out == head_dim * n_head;
    GGML_ASSERT((per_head_dim || gate_w_out == n_head) &&
            "attention output gate width matches neither n_head nor n_head*head_dim");

    auto gate = llm_build_lora_mm(lctx, ctx0, gate_w, input);
    gate = ggml_reshape_3d(ctx0, gate, per_head_dim ? head_dim : 1, n_head, output->ne[1]);
    cb(gate, "attn_gate", il);

    output = ggml_reshape_3d(ctx0, output, head_dim, n_head, output->ne[1]);
    if (per_head_dim) {
        // ggml_fused_mul_unary takes SIGMOID only on its BROADCAST path, where the gate has
        // ne[0]==1 (ggml.c:6638). With a per-head-dim gate both operands are the same shape,
        // which routes to the same-shape path -- GELU/RELU/SILU only -- so there is no fused
        // kernel for this combination. The unfused form computes the same value.
        output = ggml_mul(ctx0, output, ggml_sigmoid(ctx0, gate));
    } else {
        output = ggml_fused_mul_unary(ctx0, gate, output, GGML_UNARY_OP_SIGMOID);
    }
    output = ggml_reshape_2d(ctx0, output, head_dim * n_head, output->ne[2]);
    cb(output, "attn_gated", il);
    return output;
}

ggml_tensor * llm_build_context::llm_build_lora_mm_id(
        struct llama_context & lctx,
         struct ggml_context * ctx0,
          struct ggml_tensor * w,   // struct ggml_tensor * as
          struct ggml_tensor * cur, // struct ggml_tensor * b
          struct ggml_tensor * ids,
          struct ggml_tensor * cache_w,
          struct ggml_tensor * cache_ids) {
    struct ggml_tensor * cur_mm = cur;
    if (auto rot = lctx.model.hadamard_rotation(w); rot != nullptr) {
        // several folded weights consume the same input activation; rotate it once per graph.
        // Key by content (signs tensor + transform params), not by the per-weight transform pointer
        const auto memo_key = std::make_tuple((const struct ggml_tensor *) cur, (const struct ggml_tensor *) rot->signs,
                                              rot->block_size, rot->perm_hd, rot->perm_nk, rot->perm_rep);
        const auto memo_it  = lctx.hadamard_memo.find(memo_key);
        if (memo_it != lctx.hadamard_memo.end()) {
            cur_mm = memo_it->second;
        } else {
            cur_mm = llm_build_hadamard_rotate(ctx0, cur, *rot);
            lctx.hadamard_memo[memo_key] = cur_mm;
        }
    }
    struct ggml_tensor * res = ggml_mul_mat_id(ctx0, cache_w ? cache_w : w, cur_mm, cache_ids ? cache_ids : ids);
    for (auto & it : lctx.lora_adapters) {
        struct llama_lora_weight * lora = it.first->get_weight(w);
        if (lora == nullptr) {
            continue;
        }
        const float alpha = it.first->alpha;
        const float rank  = (float) lora->b->ne[0];
        const float scale = alpha ? it.second * alpha / rank : it.second;
        struct ggml_tensor * ab_cur = ggml_mul_mat_id(
            ctx0, lora->b,
            ggml_mul_mat_id(ctx0, lora->a, cur, ids),
            ids
        );
        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }
    if (lctx.loras) {
        for (const auto & it : *lctx.loras) {
            llama_adapter_lora_weight * lw = it.first->get_weight(w);
            if (lw == nullptr) {
                continue;
            }
            GGML_ASSERT(lw->lily_router == nullptr && "Lily on stacked experts: the loader refuses it");
            GGML_ASSERT(!lw->is_ia3() && "IA3 on stacked experts: the loader refuses it");
            struct ggml_tensor * ab_cur = ggml_mul_mat_id(ctx0, lw->b, ggml_mul_mat_id(ctx0, lw->a, cur, ids), ids);
            if (lw->dora_dm1) {
                // DoRA on stacked experts: d - 1 gathered per (slot, token) by the expert ids
                ab_cur = llama_dora_delta(ctx0, *lw, res, ggml_scale(ctx0, ab_cur, lw->get_scale(it.first->alpha, 1.0f)), ids);
                if (it.second != 1.0f) {
                    ab_cur = ggml_scale(ctx0, ab_cur, it.second);
                }
            } else {
                ab_cur = ggml_scale(ctx0, ab_cur, lw->get_scale(it.first->alpha, it.second));
            }
            if (ggml_tensor * gm = llm_lora_gate_mask(lctx, ctx0, it.first, ab_cur->ne[2])) {
                GGML_ASSERT(ab_cur->ne[3] == 1 && "gated LoRA needs [n_out, n_expert_used, n_tokens]");
                ab_cur = ggml_mul(ctx0, ab_cur, ggml_reshape_3d(ctx0, gm, 1, 1, ab_cur->ne[2]));
            }
            res = ggml_add(ctx0, res, ab_cur);
        }
    }
    for (llama_adapter_lora * ad : lctx.seq_loras.adapters) {
        llama_adapter_lora_weight * lw = ad->get_weight(w);
        if (lw == nullptr) {
            continue;
        }
        GGML_ASSERT(!lw->is_ia3() && "IA3 on stacked experts: the loader refuses it");
        struct ggml_tensor * ab_cur = ggml_mul_mat_id(ctx0, lw->b, ggml_mul_mat_id(ctx0, lw->a, cur, ids), ids);
        GGML_ASSERT(ab_cur->ne[3] == 1 && "per-sequence LoRA needs [n_out, n_expert_used, n_tokens]");
        struct ggml_tensor * mask = llm_lora_seq_mask(lctx, ctx0, ad, ab_cur->ne[2]);
        ab_cur = ggml_scale(ctx0, ab_cur, lw->get_scale(ad->alpha, 1.0f));
        if (lw->dora_dm1) {
            ab_cur = llama_dora_delta(ctx0, *lw, res, ab_cur, ids);
        }
        ab_cur = ggml_mul(ctx0, ab_cur, ggml_reshape_3d(ctx0, mask, 1, 1, ab_cur->ne[2]));
        if (ggml_tensor * gm = llm_lora_gate_mask(lctx, ctx0, ad, ab_cur->ne[2])) {
            ab_cur = ggml_mul(ctx0, ab_cur, ggml_reshape_3d(ctx0, gm, 1, 1, ab_cur->ne[2]));
        }
        res = ggml_add(ctx0, res, ab_cur);
    }
    return res;
}

ggml_tensor * llm_build_context::llm_build_norm(
        ggml_context * ctx,
         ggml_tensor * cur,
        const llama_hparams & hparams,
         ggml_tensor * mw,
         ggml_tensor * mb,
              llm_norm_type   type,
         const llm_build_cb & cb, int il, float scale_eps) {

    if (type == LLM_NORM_RMS && mw) {
        cur = ggml_fused_rms_norm(ctx, cur, mw, scale_eps * hparams.f_norm_rms_eps);
        if (mb) {
            cb(cur, "fused_norm", il);
            cur = ggml_add(ctx, cur, mb);
        }
        return cur;
    }

    switch (type) {
        case LLM_NORM:     cur = ggml_norm    (ctx, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS: cur = ggml_rms_norm(ctx, cur, scale_eps * hparams.f_norm_rms_eps); break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx, cur, mb);
    }

    return cur;
}

ggml_tensor * llm_build_context::get_input_tensor_sm_graph(ggml_context * ctx, ggml_tensor * input, int id) {
    auto cur = input;
    if (input->op == GGML_OP_REDUCE) {
        auto view_src = input->view_src;
        GGML_ASSERT(view_src);
        cur = input->src[id];
        if (!cur) {
            GGML_ASSERT((input->op_params[4] & (1u << id)) == 0);
            cur = ggml_dup_tensor(ctx, input);
            input->src[id] = cur;
            input->op_params[4] |= (1u << id);
        }
        else if (cur == view_src) {
            cur = input;
        }
    }
    return cur;
}

ggml_tensor * llm_build_context::do_split_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * the_norm, const llama_hparams & hparams,
        const llm_build_cb & cb, int id, int il_cb, bool is_norm) {
    if (the_norm && the_norm->extra) {
        auto norm = (ggml_split_tensor_t *)the_norm->extra;
        GGML_ASSERT(norm->splits[id]);
        if (is_norm) {
            cur = ggml_fused_norm(ctx, cur, norm->splits[id], hparams.f_norm_eps);
        } else {
            cur = llm_build_context::llm_build_norm(ctx, cur, hparams, norm->splits[id], NULL, LLM_NORM_RMS, cb, il_cb);
        }
        cb(cur, "inp_normed", il_cb);
    }
    if (cur->type != GGML_TYPE_F32) {
        cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
    }
    return cur;
}

static ggml_tensor * llm_do_split_post_norm(ggml_context * ctx, ggml_tensor * cur, post_norm_data * pnd, int id, int n_device, const char * tag, int il_cb, const llm_build_cb & cb) {
    auto pn_extra = (ggml_split_tensor_t *)pnd->norm->extra;
    GGML_ASSERT(pn_extra && pn_extra->splits[id]);
    GGML_ASSERT((int)pnd->next_input.size() == n_device);
    cur = ggml_fused_rms_norm(ctx, cur, pn_extra->splits[id], pnd->f_rms_eps);
    cb(cur, tag, il_cb);
    auto add = pnd->next_input[id];
    if (!add) {
        for (int j = 0; j < n_device; ++j) {
            if (pnd->next_input[j]) {
                add = pnd->next_input[j];
                break;
            }
        }
        GGML_ASSERT(add);
    }
    cur = ggml_add(ctx, cur, add);
    cb(cur, "inp_added", il_cb);
    pnd->next_input[id] = cur;
    return cur;
}

// S69 kimi-k3 "situ" activation, shared by the dense FFN and the MoE expert path so the two
// cannot drift apart. Transcribed from mainline llama.cpp's kimi_k3_situ:
//
//     situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
//
// linear_beta <= 0 disables the transform on the up branch, which is mainline's own guard and
// is kept rather than assumed impossible. beta <= 0 would divide by zero, so it falls back to
// a plain SiLU gate -- a model declaring beta 0 is malformed, but aborting the graph for it
// would be a worse failure than a defined one.
ggml_tensor * llm_build_context::llm_build_situ(ggml_context * ctx, ggml_tensor * gate,
                                    ggml_tensor * up, float beta, float linear_beta) {
    if (!(beta > 0.0f)) {
        return ggml_mul(ctx, ggml_silu(ctx, gate), up);
    }
    ggml_tensor * a = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, gate, 1.0f / beta)), beta);
    a = ggml_mul(ctx, a, ggml_sigmoid(ctx, gate));

    if (linear_beta > 0.0f) {
        up = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, up, 1.0f / linear_beta)), linear_beta);
    }
    return ggml_mul(ctx, a, up);
}

// ── CATS activation sparsity (S69-CATS-STAGE1-2026-09-28) ──────────────────────────────────
// arXiv:2404.08763 Sec 3.1, eq. 4-5: a = silu(gate(x)); a[|a| < t_l] = 0; out = down(a * up(x)).
// mask = step(|a| - t): SOURCE-READ ggml/src/ggml-cpu/unary-ops.cpp:15-17, op_step(x) returns
// 1.f for x>0.f else 0.f (strict), so mask is 1 where |a|>t and 0 where |a|<=t -- reproduces the
// paper's "< t -> 0" exactly except at the |a|==t knife-edge (zeroed here, kept there), which is
// measure-zero for a continuous activation and not tested separately. no code copied from CATS'
// own repo (not cloned; not on disk -- reference is the paper only, per the task's SOURCE line).
static ggml_tensor * cats_apply_mask(ggml_context * ctx, ggml_tensor * a, float threshold) {
    ggml_tensor * abs_a   = ggml_abs(ctx, a);                       // |a|
    ggml_tensor * shifted = ggml_scale_bias(ctx, abs_a, 1.0f, -threshold); // |a| - t
    ggml_tensor * mask    = ggml_step(ctx, shifted);                // 1 if |a|>t else 0
    return ggml_mul(ctx, a, mask);
}

// ── TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28) ──────────────────────────────────
// arXiv:2408.14690 Sec 4.2, Definition 1: stp(x)_i = 0 if |x_i| <= tp else x_i, applied to the
// INPUT of a linear projection (mask the operand BEFORE the matmul, not an activation output --
// this is why the same helper shape as CATS's mask still gives a different result: which tensor
// it is applied to differs). Note the paper's threshold test is "<= tp -> 0" (Definition 1
// literal reading) vs CATS's implementation above using ggml_step's strict "> t" for the kept
// side -- same knife-edge/measure-zero caveat as cats_apply_mask, not tested separately here.
// No code copied from the TEAL reference repo (not cloned; not on disk -- reference is the paper
// (Sec 4.2) only, per the task's SOURCE line).
ggml_tensor * teal_apply_mask(ggml_context * ctx, ggml_tensor * a, float threshold) {
    ggml_tensor * abs_a   = ggml_abs(ctx, a);
    ggml_tensor * shifted = ggml_scale_bias(ctx, abs_a, 1.0f, -threshold);
    ggml_tensor * mask    = ggml_step(ctx, shifted);
    return ggml_mul(ctx, a, mask);
}

ggml_tensor * llm_build_context::llm_build_ffn(
        ggml_context * ctx,
       llama_context & lctx,
         ggml_tensor * ffn_norm,
         ggml_tensor * input,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
            llm_ffn_op_type   type_op,
          llm_ffn_gate_type   type_gate,
         const llm_build_cb & cb, int il, ggml_cgraph * graph, bool add_input,
         bool is_norm, ggml_tensor * add_extra,
         ggml_tensor * post_norm, float post_norm_eps,
         post_norm_data * pnd) {

    // CATS activation sparsity (S69-CATS-STAGE1-2026-09-28). Only the plain gated-SiLU (SwiGLU)
    // shape the paper defines; STEP35/BAILINGMOE3 use their own fused-mul variant below and are
    // excluded here rather than forced through a path never measured for them. cats_sparsity==0
    // AND cats_calibrate==false (both default off) make cats_route_unfused false for every layer,
    // so the three fused branches below are reached exactly as before -- that is the bit-identity
    // gate. cats_calibrate routes through the same un-fused path WITHOUT masking (cats_mask_here
    // stays false) so tools/cats-calibrate can read the raw "ffn_silu" tensor on an unmodified model.
    const bool cats_eligible =
        gate && down && type_op == LLM_FFN_SILU && type_gate == LLM_FFN_PAR &&
        lctx.model.arch != LLM_ARCH_STEP35 && lctx.model.arch != LLM_ARCH_BAILINGMOE3;
    const bool cats_has_threshold =
        il >= 0 && (size_t) il < lctx.cparams.cats_thresholds.size() &&
        lctx.cparams.cats_thresholds[il] > 0.0f;
    const bool cats_route_unfused =
        cats_eligible && (lctx.cparams.cats_calibrate || (lctx.cparams.cats_sparsity > 0.0f && cats_has_threshold));
    const bool cats_mask_here =
        cats_route_unfused && !lctx.cparams.cats_calibrate && lctx.cparams.cats_sparsity > 0.0f && cats_has_threshold;

    // TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28). Input sparsity, not output-of-
    // activation like CATS -- eligible for any gate type (RELU/GELU/etc), not just SILU, since
    // Definition 1 is applied to the INPUT of Wgate/Wup/Wdown regardless of what activation
    // follows. Same arch exclusion as CATS (fused-mul variant, never measured for either).
    const bool teal_eligible =
        up && gate && down && type_gate == LLM_FFN_PAR &&
        lctx.model.arch != LLM_ARCH_STEP35 && lctx.model.arch != LLM_ARCH_BAILINGMOE3;
    const bool teal_has_ffn_in =
        il >= 0 && (size_t) il < lctx.cparams.teal_thresholds_ffn_in.size() &&
        lctx.cparams.teal_thresholds_ffn_in[il] > 0.0f;
    const bool teal_has_ffn_mid =
        il >= 0 && (size_t) il < lctx.cparams.teal_thresholds_ffn_mid.size() &&
        lctx.cparams.teal_thresholds_ffn_mid[il] > 0.0f;
    const bool teal_route_unfused =
        teal_eligible && (lctx.cparams.teal_calibrate ||
                           (lctx.cparams.teal_sparsity > 0.0f && (teal_has_ffn_in || teal_has_ffn_mid)));
    const bool teal_mask_ffn_in =
        teal_route_unfused && !lctx.cparams.teal_calibrate && lctx.cparams.teal_sparsity > 0.0f && teal_has_ffn_in;
    const bool teal_mask_ffn_mid =
        teal_route_unfused && !lctx.cparams.teal_calibrate && lctx.cparams.teal_sparsity > 0.0f && teal_has_ffn_mid;

    // Hadamard-folded up/gate weights (Prism qwen35, Gyro qwen4exp: llama_model::hadamard_rotation) need their activation
    // transform, which only the general path applies (llm_build_lora_mm). The fused up/gate nodes below take `cur` as-is,
    // so they are built only when neither weight is rotated. Upstream ik df93e4cc9 (#2812) instead rotates `cur` first and
    // fuses; that rides on ik's hadamard memo / fuse_signs plumbing, which this tree does not carry (deferred, see row
    // S69-DRIFT-1110-GYRO-TRELLIS-IQ5KS-R16-QUANT-FAMILY-2026-11-10-01). Declining is slower, never wrong.
    const bool ffn_up_gate_unrotated = !lctx.model.hadamard_rotation(up) && !lctx.model.hadamard_rotation(gate);

    if (!cats_route_unfused && !teal_route_unfused && ffn_up_gate_unrotated &&
        !up_b && !up_s && !gate_b && !gate_s && !down_b && !down_s &&
        up->extra && gate->extra && down->extra && type_gate == LLM_FFN_PAR &&
        (type_op == LLM_FFN_SILU || type_op == LLM_FFN_RELU || type_op == LLM_FFN_SWIGLU_OAI || (type_op == LLM_FFN_GELU && !act_scales))) {
        auto unary_op = type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU :
                        type_op == LLM_FFN_RELU ? GGML_UNARY_OP_RELU :
                        type_op == LLM_FFN_GELU ? GGML_UNARY_OP_GELU : GGML_UNARY_OP_SWIGLU_OAI;
        auto u = (ggml_split_tensor_t *)up->extra;
        auto g = (ggml_split_tensor_t *)gate->extra;
        auto d = (ggml_split_tensor_t *)down->extra;
        GGML_ASSERT(u->n_device == g->n_device && u->n_device == d->n_device);
        std::vector<ggml_tensor *> ffn(u->n_device, nullptr);
        int id_last = -1;
        for (int id = 0; id < u->n_device; ++id) {
            int il_cb = 1000*(id+1) + il;
            auto split_u = u->splits[id];
            auto split_g = g->splits[id];
            auto split_d = d->splits[id];
            GGML_ASSERT((!split_u && !split_g && !split_d) || (split_u && split_g && split_d));
            if (!split_u) continue;
            auto cur = get_input_tensor_sm_graph(ctx, input, id);
            if (pnd) {
                cur = llm_do_split_post_norm(ctx, cur, pnd, id, u->n_device, "attn_post_norm", il_cb, cb);
            }
            cur = do_split_norm(ctx, cur, ffn_norm, lctx.model.hparams, cb, id, il_cb, is_norm);
            if (input->op != GGML_OP_REDUCE) {
                cur->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
            }
            cur = ggml_fused_up_gate(ctx, split_u, split_g, cur, unary_op);
            cb(cur, "ffn_up_gate", il_cb);
            *(float *)(cur->op_params + 1) = lctx.model.swiglu_limit(il, lctx.model.arch == LLM_ARCH_BAILINGMOE3);
            cur = llm_build_lora_mm(lctx, ctx, split_d, cur);
            cb(cur, "ffn_down", il_cb);
            if (lctx.model.arch == LLM_ARCH_GLM4 || lctx.model.arch == LLM_ARCH_GLM4_MOE) {
                // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
                ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            }
            if (cur->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
                cur = ggml_cast(ctx, cur, lctx.cparams.reduce_type);
            }
            if (add_extra && add_extra->op == GGML_OP_REDUCE && add_extra->op_params[3] == 1 && add_extra->src[id]) {
                // When the reduce op is turned off via op_params[3] == 1, we need to add each src
                // rtaher than add the reduced add_extra result to the ffn reduced ffn result.
                cur = ggml_add(ctx, cur, add_extra->src[id]);
                cb(cur, "ffn_with_extra", il_cb);
            }
            if (graph) {
                ggml_build_forward_expand(graph, cur);
            }
            ffn[id] = cur;
            id_last = id;
        }
        GGML_ASSERT(id_last >= 0);
        if (add_input) {
            ffn[id_last] = ggml_add(ctx, ffn[id_last], input);
            cb(ffn[id_last], "ffn_with_inp", il);
        }
        if (add_extra && !(add_extra->op == GGML_OP_REDUCE && add_extra->op_params[3] == 1)) {
            ffn[id_last] = ggml_add(ctx, ffn[id_last], add_extra);
            cb(ffn[id_last], "ffn_with_inp", il);
        }
        auto cur = ggml_reduce(ctx, ffn.data(), u->n_device, GGML_OP_ADD);
        cb(cur, "ffn_combined", il);
        ggml_build_forward_expand(graph, cur);
        return cur;
    }

    if (add_extra && add_extra->op == GGML_OP_REDUCE && add_extra->op_params[3] == 1) {
        add_extra->op_params[3] = 0;
    }
    auto cur = input;
    //if (input->op == GGML_OP_REDUCE) {
    //    if (input->src[lctx.model.main_gpu]) cur = input->src[lctx.model.main_gpu];
    //}
    if (ffn_norm) {
        auto the_ffn_norm = ffn_norm->extra ? ((ggml_split_tensor_t *)ffn_norm->extra)->splits[lctx.model.main_gpu] : ffn_norm;
        cur = llm_build_norm(ctx, cur, lctx.model.hparams, the_ffn_norm, NULL, is_norm ? LLM_NORM : LLM_NORM_RMS, cb, il);
        cb(input, "ffn_norm", il);
    }
    if (cur->type != GGML_TYPE_F32) {
        cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
    }

    // TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28): mask the FFN-norm output BEFORE
    // Wgate/Wup consume it (Definition 1 input sparsity). Reached whenever teal_route_unfused
    // forced this general path, so it also covers the case where only ffn_mid has a threshold
    // this layer (teal_mask_ffn_in is then false and this is a no-op multiply-by-1 mask skip).
    cb(cur, "teal_ffn_in", il);
    if (teal_mask_ffn_in) {
        cur = teal_apply_mask(ctx, cur, lctx.cparams.teal_thresholds_ffn_in[il]);
        cb(cur, "teal_ffn_in_masked", il);
    }

    // TQ2_0 HAS NO IQK KERNEL, and the fused up/gate op has NO generic fallback:
    // ggml_compute_forward_mul_mat_up_gate ends in
    //     if (!iqk_moe_fused_up_gate(...)) GGML_ABORT("fatal error");
    // so building a fused node for a type iqk cannot handle is not a slow path, it is a hard
    // abort at ggml.c:18807. MEASURED 2026-08-20: TriLM_99M-tq2_0 loaded, allocated its KV and
    // compute buffers, printed "graph nodes = 373", and then died there on the first decode.
    // The two guards already carried with the TQ2_0 port cover plain mul_mat, where a generic
    // vec_dot fallback EXISTS; this site has none, so the fusion must not be built at all.
    if (!cats_route_unfused && !teal_route_unfused && ffn_up_gate_unrotated &&
        // TQ1_0 / TQ2_0 / Q2_0_G128, and (ik_llama.cpp #2550, MIT) the Prism ternary types
        // PQ2_0 / PQ2_0_R8 / PTQ1_0 / PTQ1_0_R8, have no fused up/gate kernel: the set is in ggml.c
        up && ggml_iqk_supports_type(up->type, GGML_IQK_SITE_GRAPH_FFN_UP_GATE) &&
        lctx.cparams.fused_up_gate &&
        up && gate && !up_b && !up_s && !gate_b && !gate_s && type_gate == LLM_FFN_PAR &&
        // a fused up/gate node has no slot for the NVFP4 per-tensor scales of up and gate: take the general path below
        !lctx.model.weight_scale(up) && !lctx.model.weight_scale(gate) &&
        // nor for a LoRA on up or gate: the fused node consumed both weights with no adapter branch, so a PEFT adapter on
        // gate_proj/up_proj loaded and changed nothing (MEASURED 2026-10-26, tests/test-adapter-peft-parity.py only_gate_proj
        // |fork-peft| 4.319 = the adapter's whole effect 4.321; only_up_proj 4.075 vs 4.076). Decided per graph build, so the
        // fusion stays on whenever no active adapter targets these weights.
        !llm_lora_targets(lctx, up) && !llm_lora_targets(lctx, gate) &&
        // ggml_fused_up_gate falls back to mul_mat x2 + ggml_fused_mul_unary for weights it cannot fuse (unquantised, or up/gate differing),
        // and that op has no SWIGLU_OAI: it asserts (ggml.c:6752, test-llama-archs -a minimax-m3, 2026-10-04). Those take the general path below.
        (type_op != LLM_FFN_SWIGLU_OAI || (ggml_is_quantized(up->type) && up->type == gate->type && ggml_are_same_shape(up, gate))) &&
        // ik 3d7b18cf0 (#2691): the fused up/gate op consumes the activation as-is; Hadamard-folded (Prism)
        // weights need their activation transform, which only llm_build_lora_mm applies -> keep them separate.
        // Covers mainline Q2_0 + prism.hadamard (Ternary-Bonsai-2), which the type exclusion above does not.
        lctx.model.hadamard_rotation(up) == nullptr && lctx.model.hadamard_rotation(gate) == nullptr &&
        (type_op == LLM_FFN_SILU || type_op == LLM_FFN_RELU || type_op == LLM_FFN_SWIGLU_OAI || (type_op == LLM_FFN_GELU && !act_scales))) {
        auto unary_op = type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU :
                        type_op == LLM_FFN_RELU ? GGML_UNARY_OP_RELU :
                        type_op == LLM_FFN_GELU ? GGML_UNARY_OP_GELU : GGML_UNARY_OP_SWIGLU_OAI;
        cur = ggml_fused_up_gate(ctx, up, gate, cur, unary_op);
        cb(cur, "ffn_up_gate", il);
        *(float *)(cur->op_params + 1) = lctx.model.swiglu_limit(il, true);
        if (down) {
            cur = llm_build_lora_mm(lctx, ctx, down, cur);
            cb(cur, "ffn_down", il);
            if (lctx.model.arch == LLM_ARCH_GLM4 || lctx.model.arch == LLM_ARCH_GLM4_MOE) {
                // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
                ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            }
        }
        if (down_b) {
            cur = ggml_add(ctx, cur, down_b);
            cb(cur, "ffn_down_b", il);
        }
        if (down_s) {
            cur = ggml_mul(ctx, cur, down_s);
            cb(cur, "ffn_down_s", il);
        }
        if (post_norm) {
            if (post_norm_eps > 0.0f) {
                cur = ggml_fused_rms_norm(ctx, cur, post_norm, post_norm_eps);
            } else {
                cur = llm_build_norm(ctx, cur, lctx.model.hparams, post_norm, NULL, LLM_NORM_RMS, cb, il);
            }
            cb(cur, "ffn_post_normed", il);
        }
        if (add_input) {
            cur = ggml_add(ctx, cur, input);
            cb(cur, "ffn_out_with_inp", il);
        }
        if (add_extra) {
            cur = ggml_add(ctx, cur, add_extra);
            cb(cur, "ffn_out_with_inp", il);
        }
        return cur;
    }

    struct ggml_tensor * tmp = up ? llm_build_lora_mm(lctx, ctx, up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = llm_build_lora_mm(lctx, ctx, gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = llm_build_lora_mm(lctx, ctx, gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    if (!cats_route_unfused && type_gate == LLM_FFN_PAR &&
       (type_op == LLM_FFN_SILU || type_op == LLM_FFN_RELU || (type_op == LLM_FFN_GELU && !act_scales))) {
        cur = ggml_fused_mul_unary(ctx, cur, tmp, type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU :
                                                  type_op == LLM_FFN_RELU ? GGML_UNARY_OP_RELU : GGML_UNARY_OP_GELU);
        *((float *)(cur->op_params + 1)) = lctx.model.swiglu_limit(il, true);
    }
    else {

    switch (type_op) {
        case LLM_FFN_SILU:
            {
                if (lctx.model.arch == LLM_ARCH_STEP35 || lctx.model.arch == LLM_ARCH_BAILINGMOE3) {
                    cur = ggml_fused_mul_unary(ctx, cur, up, GGML_UNARY_OP_SILU);
                    *(float *)(cur->op_params + 1) = lctx.model.swiglu_limit(il, true);
                    type_gate = LLM_FFN_SEQ;
                    break;
                }
                cur = ggml_silu(ctx, cur);
                cb(cur, "ffn_silu", il);
                if (cats_mask_here) {
                    cur = cats_apply_mask(ctx, cur, lctx.cparams.cats_thresholds[il]);
                    cb(cur, "ffn_cats_masked", il);
                }
            } break;
        case LLM_FFN_GELU:
            {
                cur = ggml_gelu(ctx, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            {
                cur = ggml_relu(ctx, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI:
            {
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } break;
        case LLM_FFN_SITU:
            {
                // consumes both halves itself, so the PAR multiply below must not run again
                cur = llm_build_situ(ctx, cur, tmp,
                        lctx.model.hparams.situ_beta, lctx.model.hparams.situ_linear_beta);
                cb(cur, "ffn_situ", il);
                type_gate = LLM_FFN_SEQ;
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }
    }

    // TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28): mask the gated intermediate BEFORE
    // Wdown consumes it -- this IS TEAL's Wdown input sparsity (contrast CATS's cats_mask_here
    // above, which masks silu(gate) alone, before the up-multiply; this masks the full
    // silu(gate)*up product, i.e. exactly what Wdown actually receives).
    cb(cur, "teal_ffn_mid", il);
    if (teal_mask_ffn_mid) {
        cur = teal_apply_mask(ctx, cur, lctx.cparams.teal_thresholds_ffn_mid[il]);
        cb(cur, "teal_ffn_mid_masked", il);
    }

    if (down) {
        cur = llm_build_lora_mm(lctx, ctx, down, cur);
        if (lctx.model.arch == LLM_ARCH_GLM4 || lctx.model.arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    if (post_norm) {
        if (post_norm_eps > 0.0f) {
            cur = ggml_fused_rms_norm(ctx, cur, post_norm, post_norm_eps);
        } else {
            cur = llm_build_norm(ctx, cur, lctx.model.hparams, post_norm, NULL, LLM_NORM_RMS, cb, il);
        }
        cb(cur, "ffn_post_normed", il);
    }

    if (add_input) {
        cur = ggml_add(ctx, cur, input);
        cb(cur, "ffn_out_with_inp", il);
    }
    if (add_extra) {
        cur = ggml_add(ctx, cur, add_extra);
        cb(cur, "ffn_out_with_inp", il);
    }

    return cur;
}

ggml_tensor * llm_build_context::llm_build_moe_ffn(
        ggml_context * ctx,
       llama_context & lctx,
         ggml_tensor * cur,
         ggml_tensor * gate_inp,   ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,    ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,  ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,  ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
                    int64_t   n_expert,
                    int64_t   n_expert_used,
            llm_ffn_op_type   type_op,
                       bool   norm_w,
                       bool   scale_w,
                      float   w_scale,
llm_expert_gating_func_type   gating_op,
         const llm_build_cb & cb, int il, ggml_cgraph * graph, bool add_input,
         ggml_tensor * up_gate_exps, ggml_tensor * up_gate_exps_b,
         ggml_tensor * input_logits, ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts) {

    GGML_ASSERT(gate_inp || input_logits);

    auto input = cur;

    int64_t n_embd = cur->ne[0];
    int64_t n_tokens = cur->ne[1];
    bool weight_before_ffn = lctx.model.arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    // CATS activation sparsity (S69-CATS-STAGE1-2026-09-28), MoE routed-expert FFN. Wired for
    // completeness (SILU-gated MoE is architecturally the same eq. 4-5 shape as the dense case)
    // but NOT exercised by this stage's test model (a dense SwiGLU model per the task) -- treat
    // as unverified until a MoE model is run through it. cats_sparsity==0 makes this false always.
    const bool cats_active =
        lctx.cparams.cats_sparsity > 0.0f && type_op == LLM_FFN_SILU &&
        il >= 0 && (size_t) il < lctx.cparams.cats_thresholds.size() &&
        lctx.cparams.cats_thresholds[il] > 0.0f;

    ggml_tensor * logits = gate_inp ? llm_build_lora_mm(lctx, ctx, gate_inp, cur) : input_logits; // [n_expert, n_tokens]
    cb(logits, "ffn_moe_logits", il);

    if (gate_inp_b) {
        logits = ggml_add(ctx, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    if (gating_op == LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    }

    //ggml_tensor * probs = ggml_soft_max(ctx, logits); // [n_expert, n_tokens]
    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLM_EXPERT_GATING_FUNC_SOFTMAX:
            {
                probs = ggml_soft_max(ctx, logits); // [n_expert, n_tokens]
            } break;
        case LLM_EXPERT_GATING_FUNC_SIGMOID:
            {
                probs = ggml_sigmoid(ctx, logits); // [n_expert, n_tokens]
            } break;
        case LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                //probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits)); // [n_expert, n_tokens]
                probs = ggml_sqrt_softplus(ctx, logits); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (lctx.model.arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    // select experts
    if (selected_experts == nullptr) {
        const bool grouped_routing = lctx.cparams.grouped_expert_routing &&
                (lctx.model.arch == LLM_ARCH_BAILINGMOE2 || lctx.model.arch == LLM_ARCH_BAILINGMOE3);
        if (grouped_routing && n_tokens > 0) {
            auto& hparams = lctx.model.hparams;
            selected_experts = ggml_grouped_topk(ctx, selection_probs, hparams.n_expert_groups, hparams.n_group_used, 2, n_expert_used);
        } else if (lctx.cparams.min_experts > 0 && lctx.cparams.thresh_experts > 0.0f &&
                   (up_exps_b || gate_exps_b || down_exps_b || down_exps_s)) {
            // ggml_add_id (expert biases) and the per-expert down scales (ggml_mul_multi_add src[3]) are not SER-aware:
            // add_id asserts on an id of -1 on CPU. Say so once, run dense top-k.
            static bool warned = false;
            if (!warned) {
                warned = true;
                LLAMA_LOG_WARN("%s: -ser is ignored for MoE layers with expert biases or per-expert scales (ggml_add_id / the scaled multi-add cannot skip an expert); using plain top-k\n", __func__);
            }
            selected_experts = ggml_argsort_top_k(ctx, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        } else if (lctx.cparams.min_experts > 0 && lctx.cparams.thresh_experts > 0.0f) {
            // SER (-ser N,T / --smart-expert-reduction): past the first N experts of the top-k, an expert whose router
            // score is below T * (best score) gets id -1. mul_mat_id, the fused up-gate and get_rows (CPU ops.cpp,
            // CUDA getrows.cu) treat -1 as "skip / zero row", so the expert contributes nothing and norm_w renormalises.
            // The call was commented out in ik_llama's own refactor (4daff01b3, still so in portable/ik_llama_src
            // src/llama-build-context.cpp:1083) and in this fork's pre-merge tree (35f0cac75 :1586), leaving -ser
            // accepted, logged and read by nothing (row S69-FLAG-WIRING-SER-NO-READER-2026-10-01-01).
            // Default (-1, 0) and min_experts == 0 keep the plain top-k below, bit-identical to before.
            selected_experts = ggml_top_k_thresh(ctx, selection_probs, n_expert_used,
                    lctx.cparams.min_experts, lctx.cparams.thresh_experts); // [n_expert_used, n_tokens]
        } else {
            selected_experts = ggml_argsort_top_k(ctx, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        }
    }

    cb(selected_experts, "ffn_moe_topk", il);
    ggml_tensor * weights = ggml_get_rows(ctx,
            ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);

    if (gating_op == LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max_inplace(ctx, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        if (lctx.model.arch == LLM_ARCH_LAGUNA) {
            weights_sum = ggml_clamp(ctx, weights_sum, 6.103515625e-5f, INFINITY);
            cb(weights_sum, "ffn_moe_weights_sum_clamped", il);
        } else if (lctx.model.arch == LLM_ARCH_BAILINGMOE2 || lctx.model.arch == LLM_ARCH_BAILINGMOE3 || lctx.model.arch == LLM_ARCH_STEP35) {
            weights_sum = ggml_scale_bias(ctx, weights_sum, 1.0, 1e-20);
            cb(weights_sum, "ffn_moe_weights_sum_biased", il);
        }

        weights = ggml_div(ctx, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    }
    if (scale_w && std::abs(w_scale-1) > 1e-5f) {
        weights = ggml_scale(ctx, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    if (graph) {
        ggml_build_forward_expand(graph, weights);
    }

    ggml_tensor * cache_ids = nullptr;
    ggml_tensor * cached_up_gate = nullptr;
    ggml_tensor * cached_up = nullptr;
    ggml_tensor * cached_gate = nullptr;
    ggml_tensor * cached_down = nullptr;
    ggml_tensor * cpu_ids = nullptr; // the misses the CPU computes from the host experts (llama_moe_cache::get_cpu_map), -1 for the other experts
    // The cache substitutes its own copies of the expert stacks, which carry no LoRA: skip it only when an active adapter (legacy map,
    // context-wide loras, per-sequence seq_loras: llm_lora_targets checks all three) targets one of THIS layer's expert stacks. The old
    // test, lora_adapters.empty(), read the legacy map only, so a server/--lora adapter on the experts was silently dropped here (row
    // S69-ENGREV-PEFT-LORA-FASTPATH-GUARD-LEGACY-MAP-ONLY-2026-10-27-01).
    const bool moe_cache_adapted = llm_lora_targets(lctx, up_gate_exps) || llm_lora_targets(lctx, up_exps) ||
                                   llm_lora_targets(lctx, gate_exps) || llm_lora_targets(lctx, down_exps);
    if (lctx.moe_cache && !moe_cache_adapted) {
        cached_up_gate = lctx.moe_cache->get_experts(up_gate_exps);
        cached_up = lctx.moe_cache->get_experts(up_exps);
        cached_gate = lctx.moe_cache->get_experts(gate_exps);
        cached_down = lctx.moe_cache->get_experts(down_exps);
        const bool complete = (up_gate_exps ? cached_up_gate != nullptr :
                (up_exps ? cached_up != nullptr : true) && (gate_exps ? cached_gate != nullptr : true)) &&
                (down_exps ? cached_down != nullptr : true);
        // upstream llama.cpp e95426f16 (#30966): part of the misses runs on the CPU from the host experts. Not with weights before the FFN, LoRAs,
        // rotated experts or expert biases (the CPU reads id -1 as "skip", the biases are indexed by id), and it needs the graph to order the
        // CPU part after the cached part
        ggml_tensor * cpu_map = nullptr;
        if (complete && graph && !weight_before_ffn && (!lctx.loras || lctx.loras->empty()) && lctx.seq_loras.adapters.empty() &&
                !up_exps_b && !gate_exps_b && !down_exps_b && !up_gate_exps_b &&
                lctx.model.hadamard_rotation(up_exps) == nullptr && lctx.model.hadamard_rotation(gate_exps) == nullptr &&
                lctx.model.hadamard_rotation(up_gate_exps) == nullptr && lctx.model.hadamard_rotation(down_exps) == nullptr) {
            cpu_map = lctx.moe_cache->get_cpu_map(il, n_tokens);
        }
        ggml_tensor * slot_map = lctx.moe_cache->get_slot_map(il, n_tokens, n_expert_used, cpu_map != nullptr);
        if (cpu_map && slot_map) {
            // compute the expert input before the slot lookup, so that the scheduler prefetches it for the CPU part (upstream cf34d3113 #31888)
            ggml_build_forward_expand(graph, cur);
        }
        if (slot_map && complete) {
            ggml_tensor * cache_lookup_ids = ggml_is_contiguous(selected_experts) ? selected_experts :
                    ggml_cont(ctx, selected_experts);
            cache_lookup_ids = ggml_reshape_1d(ctx, cache_lookup_ids, ggml_nelements(cache_lookup_ids));
            ggml_tensor * cache_slot_ids = ggml_get_rows(ctx, slot_map, cache_lookup_ids);
            ggml_backend_t backend = lctx.moe_cache->backend(il);
            if (ggml_backend_supports_op(backend, cache_slot_ids)) {
                ggml_backend_sched_set_tensor_backend(lctx.sched, cache_slot_ids, backend);
                cache_ids = ggml_reshape_2d(ctx, cache_slot_ids, n_expert_used, n_tokens);
                if (cpu_map) {
                    // the lookup comes first in the graph: the slot map copy plans the layer, and cpu_map is read after it. cpu_map is a host weight,
                    // so this get_rows runs on the CPU
                    ggml_build_forward_expand(graph, cache_slot_ids);
                    cpu_ids = ggml_reshape_2d(ctx, ggml_get_rows(ctx, cpu_map, cache_lookup_ids), n_expert_used, n_tokens);
                }
            } else {
                cached_up_gate = cached_up = cached_gate = cached_down = nullptr;
            }
        } else {
            cached_up_gate = cached_up = cached_gate = cached_down = nullptr;
        }
    }

    cur = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // TODO: this is a workaround as we don't yet have a repeat op that takes custom dim (ggml_repeat_4d)
        ggml_tensor * repeated = ggml_new_tensor_3d(ctx, cur->type, n_embd, n_expert_used, n_tokens);
        repeated = ggml_repeat(ctx, cur, repeated); // [n_embd, n_expert_used, n_tokens]
        cur = ggml_mul(ctx, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    // For now we don't modify the fused up/gate op to include biases.
    // Hence, if we have biases, we cannot use fmoe.
    //
    //bool can_use_fmoe = !up_exps_b && !gate_exps_b && (type_op == LLM_FFN_SILU || type_op == LLM_FFN_GELU);
    // Same reason as the dense site above: the fused MoE up/gate op aborts when iqk has no
    // kernel for the type. Both branches below have unfused else-paths, so declining the
    // fusion here is a slower graph rather than a missing capability.
    const enum ggml_type fmoe_type = up_gate_exps ? up_gate_exps->type
                                   : (up_exps ? up_exps->type : GGML_TYPE_F32);
    // The fused ggml_moe_up_gate takes the expert stacks only, so an adapter on up/gate experts would be silently dropped
    // (same defect as the dense site, row S69-ADAPTERS-FUSED-UP-GATE-DROPS-LORA-2026-10-26-01; here for the MoE path, row
    // S69-ADAPTERS-W2-MULTI-DORA-BATCH-AND-MOE-2026-10-26-01): an adapted stack takes the unfused llm_build_lora_mm_id path.
    const bool moe_up_gate_adapted = (up_gate_exps && llm_lora_targets(lctx, up_gate_exps)) ||
                                     (up_exps && llm_lora_targets(lctx, up_exps)) || (gate_exps && llm_lora_targets(lctx, gate_exps));
    bool can_use_fmoe = !cats_active && !moe_up_gate_adapted
                        && (type_op == LLM_FFN_SILU || type_op == LLM_FFN_GELU || type_op == LLM_FFN_SWIGLU_OAI)
                        // TQ1_0 / TQ2_0 / Q2_0_G128 out (the set is in ggml.c)
                        && ggml_iqk_supports_type(fmoe_type, GGML_IQK_SITE_GRAPH_MOE_UP_GATE);

    // cpu_pass: the host experts selected by cache_ids = cpu_ids (negative = skipped, those rows read as zero); the cached part passes the
    // bank views and the slots. The two results are added before the down bias. The parameters shadow the cache variables above on purpose.
    auto build_experts = [&](ggml_tensor * cached_up_gate, ggml_tensor * cached_up, ggml_tensor * cached_gate, ggml_tensor * cached_down,
            ggml_tensor * cache_ids, bool cpu_pass) {
    ggml_tensor * par;
    if (can_use_fmoe && up_gate_exps && (!cache_ids || !up_gate_exps_b)) {
        // ik_llama 35e0be560 (#2822): the fused op takes the merged up/gate stack and the raw activations, so a Prism Hadamard
        // rotation configured for these experts has to be applied to the activations here (the unfused branch below gets it from
        // llm_build_lora_mm_id). The fork keys the rotation by tensor name (hadamard_map) and a merged stack may carry its own
        // key or only its up / gate halves' keys: take the merged key first, else the halves', and require the halves to agree.
        {
            const llama_hadamard_transform * had = lctx.model.hadamard_rotation(up_gate_exps);
            if (!had) {
                const llama_hadamard_transform * had_u = up_exps   ? lctx.model.hadamard_rotation(up_exps)   : nullptr;
                const llama_hadamard_transform * had_g = gate_exps ? lctx.model.hadamard_rotation(gate_exps) : nullptr;
                GGML_ASSERT((!had_u && !had_g) || (had_u == had_g) ||
                        (had_u && had_g && llm_hadamard_same(*had_u, *had_g)));
                had = had_u ? had_u : had_g;
            }
            if (had) {
                cur = llm_build_hadamard_rotate(ctx, cur, *had);
            }
        }
        ggml_tensor * weight = cached_up_gate ? cached_up_gate : up_gate_exps;
        ggml_tensor * expert_ids = cache_ids ? cache_ids : selected_experts;
        if (up_gate_exps_b || type_op == LLM_FFN_SWIGLU_OAI) {
            par = ggml_moe_up_gate_ext(ctx, weight, nullptr, cur, expert_ids, up_gate_exps_b, nullptr,
                    type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU :
                    type_op == LLM_FFN_GELU ? GGML_UNARY_OP_GELU : GGML_UNARY_OP_SWIGLU_OAI);
        } else {
            GGML_ASSERT(type_op != LLM_FFN_SWIGLU_OAI);
            par = ggml_moe_up_gate(ctx, weight, nullptr, cur, expert_ids,
                    type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU : GGML_UNARY_OP_GELU);
        }
        *((float *)(par->op_params + 1)) = lctx.model.swiglu_limit(il, false);
    } else {
    if (up_gate_exps) {
        GGML_ASSERT((cache_ids && (cached_up_gate || cpu_pass)) || moe_up_gate_adapted);
        ggml_tensor * gate_up = llm_build_lora_mm_id(lctx, ctx, up_gate_exps, cur,
                selected_experts, cached_up_gate, cache_ids);
        if (up_gate_exps_b) {
            gate_up = ggml_add_id(ctx, gate_up, up_gate_exps_b, selected_experts);
        }
        const int64_t n_ff = gate_up->ne[0] / 2;
        ggml_tensor * gate = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2],
                gate_up->nb[1], gate_up->nb[2], 0);
        ggml_tensor * up = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2],
                gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        if (type_op == LLM_FFN_SILU || type_op == LLM_FFN_GELU) {
            par = ggml_fused_mul_unary(ctx, gate, up,
                    type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU : GGML_UNARY_OP_GELU);
            *((float *)(par->op_params + 1)) = lctx.model.swiglu_limit(il, false);
        } else if (type_op == LLM_FFN_SWIGLU_OAI) {
            par = ggml_swiglu_oai(ctx, gate, up, 1.702f, 7.0f);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
    GGML_ASSERT(!up_gate_exps_b);

    // same rule as llm_build_ffn: a fused MoE up/gate node takes `cur` un-rotated, so rotated experts take the
    // general path (llm_build_lora_mm_id applies the transform); looked up on the original expert stacks, by name
    if (can_use_fmoe && lctx.cparams.fused_moe_up_gate && up_exps->type == gate_exps->type &&
            lctx.model.hadamard_rotation(up_exps) == nullptr && lctx.model.hadamard_rotation(gate_exps) == nullptr &&
            (!cache_ids || (!up_exps_b && !gate_exps_b))) {
        ggml_tensor * up_weight = cached_up ? cached_up : up_exps;
        ggml_tensor * gate_weight = cached_gate ? cached_gate : gate_exps;
        ggml_tensor * expert_ids = cache_ids ? cache_ids : selected_experts;
        if (up_exps_b || gate_exps_b || type_op == LLM_FFN_SWIGLU_OAI) {
            par = ggml_moe_up_gate_ext(ctx, up_weight, gate_weight, cur, expert_ids, up_exps_b, gate_exps_b,
                    type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU :
                    type_op == LLM_FFN_GELU ? GGML_UNARY_OP_GELU : GGML_UNARY_OP_SWIGLU_OAI);
        } else {
            GGML_ASSERT(type_op != LLM_FFN_SWIGLU_OAI);
            par = ggml_moe_up_gate(ctx, up_weight, gate_weight, cur, expert_ids,
                    type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU : GGML_UNARY_OP_GELU);
        }
        *(float *)(par->op_params + 1) = lctx.model.swiglu_limit(il, false);
    } else {
        ggml_tensor * up = llm_build_lora_mm_id(lctx, ctx, up_exps, cur, selected_experts, cached_up, cache_ids); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        ggml_tensor * gate = llm_build_lora_mm_id(lctx, ctx, gate_exps, cur, selected_experts, cached_gate, cache_ids); // [n_ff, n_expert_used, n_tokens]
        cb(gate, "ffn_moe_gate", il);

        if (graph) {
            // So we can potentially fuse the up and gate mul_mat_id
            ggml_build_forward_expand(graph, up);
            ggml_build_forward_expand(graph, gate);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps_b) {
            gate = ggml_add_id(ctx, gate, gate_exps_b, selected_experts);
            cb(gate, "ffn_moe_gate_biased", il);
        }

        if (cats_active && type_op == LLM_FFN_SILU) {
            ggml_tensor * act = ggml_silu(ctx, gate);
            cb(act, "ffn_moe_silu", il);
            act = cats_apply_mask(ctx, act, lctx.cparams.cats_thresholds[il]);
            cb(act, "ffn_moe_cats_masked", il);
            par = ggml_mul(ctx, act, up);
            cb(par, "ffn_moe_gate_par_masked", il);
        } else if (type_op == LLM_FFN_SILU || type_op == LLM_FFN_GELU) {
            par = ggml_fused_mul_unary(ctx, gate, up, type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU : GGML_UNARY_OP_GELU);
            *((float *)(par->op_params + 1)) = lctx.model.swiglu_limit(il, false);
        } else if (type_op == LLM_FFN_SWIGLU_OAI) {
            constexpr float alpha = 1.702f;
            constexpr float limit = 7.0f;
            par = ggml_swiglu_oai(ctx, gate, up, alpha, limit);
        }
        else if (type_op == LLM_FFN_SITU) {
            // kimi-k3 routed experts. can_use_fmoe lists SILU/GELU/SWIGLU_OAI only, so SITU
            // reaches this unfused branch by construction and shares llm_build_situ with the
            // dense path above -- one definition of the activation, two callers.
            par = llm_build_situ(ctx, gate, up,
                    lctx.model.hparams.situ_beta, lctx.model.hparams.situ_linear_beta);
        }
        else {
            GGML_ABORT("fatal error");
        }

    }
    }
    }
    cb(par, "ffn_moe_gate_par", il);

    ggml_tensor * experts = llm_build_lora_mm_id(lctx, ctx, down_exps, par, selected_experts, cached_down, cache_ids); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);
    return experts;
    }; // build_experts

    ggml_tensor * experts = build_experts(cached_up_gate, cached_up, cached_gate, cached_down, cache_ids, false);
    if (cpu_ids != nullptr) {
        // the cached part comes first in the graph, so that it is computed while the CPU computes the misses
        ggml_build_forward_expand(graph, experts);
        experts = ggml_add(ctx, experts, build_experts(nullptr, nullptr, nullptr, nullptr, cpu_ids, true));
        cb(experts, "ffn_moe_down_merged", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (down_exps_s && !lctx.cparams.fused_mmad) {
        GGML_ASSERT(!weight_before_ffn);
        auto s = ggml_reshape_3d(ctx, down_exps_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx, s, selected_experts);
        auto w_reshaped = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
        auto s_reshaped = ggml_reshape_2d(ctx, s, n_expert_used, n_tokens);
        w_reshaped = ggml_mul(ctx, w_reshaped, s_reshaped);
        weights = ggml_reshape_3d(ctx, w_reshaped, 1, n_expert_used, n_tokens);
    }

    if (!weight_before_ffn) {
        if (lctx.cparams.fused_mmad) {
            experts = ggml_mul_multi_add(ctx, experts, weights);
            cb(experts, "ffn_moe_weighted", il);
            if (down_exps_s) {
                experts->src[2] = down_exps_s;
                experts->src[3] = selected_experts;
            }
            if (add_input) {
                experts = ggml_add(ctx, experts, input);
                cb(experts, "ffn_out_with_inp", il);
            }
            return experts;
        }
        experts = ggml_mul(ctx, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_tensor * result;
    if (n_expert_used == 1) {
        result = ggml_cont(ctx, ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0));
    }
    if (n_expert_used == 2) {
        result = ggml_add(ctx, ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0),
                             ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], experts->nb[1]));
    }
    result = ggml_multi_add(ctx, ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0), n_expert_used);
    if (add_input) {
        cb(result, "ffn_out", il);
        result = ggml_add(ctx, result, input);
    }
    return result;

}

ggml_tensor * llm_build_context::llm_build_std_moe_ffn(ggml_context * ctx, llama_context & lctx,
         ggml_tensor * ffn_norm,
         ggml_tensor * input,
         ggml_tensor * gate_inp,   ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,    ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,  ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,  ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
         ggml_tensor * up_shexp,   ggml_tensor * up_b_shexp,
         ggml_tensor * gate_shexp, ggml_tensor * gate_b_shexp,
         ggml_tensor * down_shexp, ggml_tensor * down_b_shexp,
                    int64_t   n_expert,
                    int64_t   n_expert_used,
            llm_ffn_op_type   type_op,
                       bool   norm_w,
                       bool   scale_w,
                      float   w_scale,
llm_expert_gating_func_type   gating_op,
            llm_ffn_op_type   type_op_shexp,
         const llm_build_cb & cb, int il, ggml_cgraph * graph, bool add_input,
         ggml_tensor * up_gate_exps, ggml_tensor * up_gate_exps_b,
         ggml_tensor * shexp_gate,
         ggml_tensor * add_extra) {

    auto split_up_exps    = up_exps ? (ggml_split_tensor_t *)up_exps->extra : nullptr;
    auto split_gate_exps  = gate_exps ? (ggml_split_tensor_t *)gate_exps->extra : nullptr;
    auto split_down_exps  = (ggml_split_tensor_t *)down_exps->extra;
    auto split_up_shexp   = up_shexp   ? (ggml_split_tensor_t *)up_shexp->extra   : nullptr;
    auto split_gate_shexp = gate_shexp ? (ggml_split_tensor_t *)gate_shexp->extra : nullptr;
    auto split_down_shexp = down_shexp ? (ggml_split_tensor_t *)down_shexp->extra : nullptr;
    auto split_up_b_shexp   = up_b_shexp   ? (ggml_split_tensor_t *)up_b_shexp   : nullptr;
    auto split_gate_b_shexp = gate_b_shexp ? (ggml_split_tensor_t *)gate_b_shexp : nullptr;
    auto split_down_b_shexp = down_b_shexp ? (ggml_split_tensor_t *)down_b_shexp : nullptr;
    auto split_up_gate_exps = up_gate_exps ? (ggml_split_tensor_t *)up_gate_exps->extra : nullptr;
    if (!split_up_exps && !split_gate_exps && !split_up_gate_exps && !split_down_exps) {
        auto cur = input;
        if (ffn_norm) {
            auto the_ffn_norm = ffn_norm->extra ? ((ggml_split_tensor_t *)ffn_norm->extra)->splits[lctx.model.main_gpu] : ffn_norm;
            GGML_ASSERT(the_ffn_norm);
            cur = llm_build_norm(ctx, cur, lctx.model.hparams, the_ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_inp_normed", il);
        }
        if (cur->type != GGML_TYPE_F32) {
            cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
        }
        auto the_gate_inp = gate_inp->extra ? ((ggml_split_tensor_t *)gate_inp->extra)->splits[lctx.model.main_gpu] : gate_inp;
        auto the_gate_inp_b = gate_inp_b ? gate_inp_b->extra ? ((ggml_split_tensor_t *)gate_inp_b->extra)->splits[lctx.model.main_gpu] : gate_inp_b : nullptr;
        auto the_exp_probs_b = exp_probs_b ? exp_probs_b->extra ? ((ggml_split_tensor_t *)exp_probs_b->extra)->splits[lctx.model.main_gpu] : exp_probs_b : nullptr;
        //int n_before = graph->n_nodes;
        auto routed_out = llm_build_moe_ffn(ctx, lctx, cur,
                    the_gate_inp, the_gate_inp_b,
                    up_exps,   up_exps_b,
                    gate_exps, gate_exps_b,
                    down_exps, down_exps_b,
                    the_exp_probs_b,
                    n_expert, n_expert_used,
                    type_op, norm_w, scale_w, w_scale,
                    gating_op, cb, il, graph, false, up_gate_exps, up_gate_exps_b);
        cb(routed_out, "routed_out", il);
        if (add_input) {
            routed_out = ggml_add(ctx, routed_out, input);
            cb(routed_out, "routed_out_with_inp", il);
        }
        ggml_build_forward_expand(graph, routed_out);

        bool handled_add_extra = false;
        if (up_shexp && gate_shexp && down_shexp) {
            if (split_up_shexp) {
                std::vector<ggml_tensor *> results(split_up_shexp->n_device, nullptr);
                GGML_ASSERT(!split_up_b_shexp   || split_up_b_shexp->n_device   == split_up_shexp->n_device);
                GGML_ASSERT(!split_gate_b_shexp || split_gate_b_shexp->n_device == split_up_shexp->n_device);
                GGML_ASSERT(!split_down_b_shexp || split_down_b_shexp->n_device == split_up_shexp->n_device);
                bool down_bias_added = false;
                int id_add_routed = -1;
                if (split_up_shexp->splits[lctx.model.main_gpu]) {
                    id_add_routed = lctx.model.main_gpu;
                } else {
                    for (int id = 0; id < split_up_shexp->n_device; ++id) {
                        if (split_up_shexp->splits[id]) {
                            id_add_routed = id;
                            break;
                        }
                    }
                }
                GGML_ASSERT(id_add_routed >= 0);
                for (int id = 0; id < split_up_shexp->n_device; ++id) {
                    int il_cb = 1000*id + il;
                    GGML_ASSERT((split_up_shexp->splits[id] && split_gate_shexp->splits[id] && split_down_shexp->splits[id]) ||
                                (!split_up_shexp->splits[id] && !split_gate_shexp->splits[id] && !split_down_shexp->splits[id]));
                    if (!split_up_shexp->splits[id]) continue;
                    auto the_ffn_norm = ffn_norm ? ffn_norm->extra ? ((ggml_split_tensor_t *)ffn_norm->extra)->splits[id] : ffn_norm : nullptr;
                    auto this_input = input;
                    if (the_ffn_norm) {
                        this_input = llm_build_norm(ctx, input, lctx.model.hparams, the_ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
                    }
                    auto shared_out = llm_build_ffn(ctx, lctx, nullptr, this_input,
                            split_up_shexp->splits[id],   split_up_b_shexp   ? split_up_b_shexp->splits[id]   : nullptr, nullptr,
                            split_gate_shexp->splits[id], split_gate_b_shexp ? split_gate_b_shexp->splits[id] : nullptr, nullptr,
                            split_down_shexp->splits[id], !down_bias_added && split_down_b_shexp ? split_down_b_shexp->splits[id] : nullptr, nullptr,
                            nullptr, type_op_shexp, LLM_FFN_PAR, cb, il, graph, false, false, nullptr);
                    cb(shared_out, "ffn_shexp_out", il_cb);
                    if (shexp_gate) {
                        auto split_shexp_gate = (ggml_split_tensor_t *)shexp_gate->extra;
                        GGML_ASSERT(split_shexp_gate && split_shexp_gate->splits[id]);
                        auto gate = llm_build_lora_mm(lctx, ctx, split_shexp_gate->splits[id], this_input);
                        // one op for every ubatch size: sigmoid + mul for a window only gave a T-token window and T single decodes different
                        // state (test-spec-window-parity). Row S69-FORK-SHEXP-GATE-BATCH-VARIANT-SIGMOID-BRANCH-2026-11-05-01
                        shared_out = ggml_fused_mul_unary(ctx, gate, shared_out, GGML_UNARY_OP_SIGMOID);
                        cb(shared_out, "ffn_shexp_gated", il_cb);
                    }
                    if (id == id_add_routed) {
                        shared_out = ggml_add(ctx, shared_out, routed_out);
                        cb(shared_out, "ffn_shared_routed_added", il);
                    }
                    if (add_extra && add_extra->op == GGML_OP_REDUCE && add_extra->op_params[3] == 1) {
                        GGML_ASSERT(add_extra->src[id]); // TODO: fix this! It can be null if the splits of the attention and ffn tensors are different
                        shared_out = ggml_add(ctx, shared_out, add_extra->src[id]);
                        cb(shared_out, "ffn_shared_with_extra", il_cb);
                        handled_add_extra = true;
                    }
                    if (shared_out->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
                        shared_out = ggml_cast(ctx, shared_out, lctx.cparams.reduce_type);
                    }
                    ggml_build_forward_expand(graph, shared_out);
                    down_bias_added = true;
                    results[id] = shared_out;
                }
                GGML_ASSERT(!results.empty());
                cur = ggml_reduce(ctx, results.data(), split_up_shexp->n_device, GGML_OP_ADD);
                cb(cur, "ffn_out", il);
            } else {
                auto shared_out = llm_build_ffn(ctx, lctx, nullptr, cur,
                        up_shexp,   up_b_shexp,   nullptr,
                        gate_shexp, gate_b_shexp, nullptr,
                        down_shexp, down_b_shexp, nullptr,
                        nullptr, type_op_shexp, LLM_FFN_PAR, cb, il, graph);
                cb(shared_out, "ffn_shexp_out", il);
                if (shexp_gate) {
                    auto shared_gate = llm_build_lora_mm(lctx, ctx, shexp_gate, cur);
                    cb(shared_gate, "shared_expert_gate", il);
                    // one op for every ubatch size: sigmoid + mul for a window only gave a T-token window and T single decodes different
                    // state (test-spec-window-parity). Row S69-FORK-SHEXP-GATE-BATCH-VARIANT-SIGMOID-BRANCH-2026-11-05-01
                    shared_out = ggml_fused_mul_unary(ctx, shared_gate, shared_out, GGML_UNARY_OP_SIGMOID);
                    cb(shared_out, "ffn_shexp_gated", il);
                }
                cur = ggml_add(ctx, routed_out, shared_out);
                cb(cur, "ffn_out", il);
            }
        } else {
            cur = routed_out;
        }
        if (add_extra && !handled_add_extra) {
            if (add_extra->op == GGML_OP_REDUCE && add_extra->op_params[3] == 1) {
                add_extra->op_params[3] = 0;
            }
            cur = ggml_add(ctx, cur, add_extra);
            cb(cur, "ffn_with_extra", il);
        }
        if (cur != routed_out) {
            ggml_build_forward_expand(graph, cur);
        }
        return cur;
    }
    GGML_ASSERT(((split_up_exps && split_gate_exps) || split_up_gate_exps) && split_down_exps);
    int n_device = split_down_exps->n_device;
    if (split_up_gate_exps) {
        GGML_ASSERT(split_up_gate_exps->n_device == n_device);
    } else {
        GGML_ASSERT(split_up_exps->n_device == n_device && split_gate_exps->n_device == n_device);
    }
    std::vector<ggml_tensor *> results(n_device, nullptr);
    GGML_ASSERT((!split_up_shexp && !split_gate_shexp && !split_down_shexp) ||
                ( split_up_shexp &&  split_gate_shexp &&  split_down_shexp));
    auto split_gate_inp = (ggml_split_tensor_t *)gate_inp->extra;
    GGML_ASSERT(split_gate_inp && split_gate_inp->n_device == n_device);
    auto split_exp_probs_b = exp_probs_b ? (ggml_split_tensor_t *)exp_probs_b->extra : nullptr;
    GGML_ASSERT(!split_exp_probs_b || split_exp_probs_b->n_device == n_device);

    auto split_gate_inp_b  = gate_inp_b  ? (ggml_split_tensor_t *)gate_inp_b->extra  : nullptr;
    auto split_exps_down_b = down_exps_b ? (ggml_split_tensor_t *)down_exps_b->extra : nullptr;
    auto split_exps_gate_b = gate_exps_b ? (ggml_split_tensor_t *)gate_exps_b->extra : nullptr;
    auto split_exps_up_b   = up_exps_b   ? (ggml_split_tensor_t *)up_exps_b->extra   : nullptr;
    auto split_exps_up_gate_b = up_gate_exps_b ? (ggml_split_tensor_t *)up_gate_exps_b->extra : nullptr;
    int last_id = -1;
    bool down_bias_added = false;
    for (int id = 0; id < n_device; ++id) {
        bool has_up_gate = split_up_gate_exps ? split_up_gate_exps->splits[id] != nullptr : split_up_exps->splits[id] != nullptr && split_gate_exps->splits[id]!= nullptr ;
        GGML_ASSERT((has_up_gate && split_down_exps->splits[id]) ||
                    (!has_up_gate && !split_down_exps->splits[id]));
        if (!has_up_gate) continue;
        int il_cb = 1000*(id + 1) + il;
        auto cur = get_input_tensor_sm_graph(ctx, input, id);
        cur = do_split_norm(ctx, cur, ffn_norm, lctx.model.hparams, cb, id, il_cb, false);
        if (cur->op != GGML_OP_REDUCE) {
            cur->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
        }
        GGML_ASSERT(!split_gate_inp_b  || split_gate_inp_b->splits[id]);
        GGML_ASSERT(!split_exps_down_b || split_exps_down_b->splits[id]);
        GGML_ASSERT(!split_exps_gate_b || split_exps_gate_b->splits[id]);
        GGML_ASSERT(!split_exps_up_b   || split_exps_up_b->splits[id]);
        auto routed_out = llm_build_moe_ffn(ctx, lctx, cur,
                    split_gate_inp->splits[id],  split_gate_inp_b ? split_gate_inp_b->splits[id] : nullptr,
                    split_up_exps ? split_up_exps->splits[id] : nullptr, split_exps_up_b  ? split_exps_up_b->splits[id]  : nullptr,
                    split_gate_exps ? split_gate_exps->splits[id] : nullptr, split_exps_gate_b ? split_exps_gate_b->splits[id] : nullptr,
                    split_down_exps->splits[id], !down_bias_added && split_exps_down_b ? split_exps_down_b->splits[id] : nullptr,
                    split_exp_probs_b ? split_exp_probs_b->splits[id] : nullptr,
                    n_expert, n_expert_used,
                    type_op, norm_w, scale_w, w_scale,
                    gating_op, cb, il, graph, false,
                    split_up_gate_exps ? split_up_gate_exps->splits[id] : nullptr,
                    split_exps_up_gate_b ? split_exps_up_gate_b->splits[id] : nullptr);
        cb(routed_out, "routed_out", il_cb);

        if (split_up_shexp) {
            GGML_ASSERT(!split_up_b_shexp   || split_up_b_shexp->n_device   == n_device);
            GGML_ASSERT(!split_gate_b_shexp || split_gate_b_shexp->n_device == n_device);
            GGML_ASSERT(!split_down_b_shexp || split_down_b_shexp->n_device == n_device);
            auto shared_out = llm_build_ffn(ctx, lctx, nullptr, cur,
                    split_up_shexp->splits[id],   split_up_b_shexp   ? split_up_b_shexp->splits[id]   : nullptr, nullptr,
                    split_gate_shexp->splits[id], split_gate_b_shexp ? split_gate_b_shexp->splits[id] : nullptr, nullptr,
                    split_down_shexp->splits[id], !down_bias_added && split_down_b_shexp ? split_down_b_shexp->splits[id] : nullptr, nullptr,
                    nullptr, type_op_shexp, LLM_FFN_PAR, cb, il);
            cb(shared_out, "ffn_shexp_out", il_cb);
            if (shexp_gate) {
                auto split_shexp_gate = (ggml_split_tensor_t *)shexp_gate->extra;
                GGML_ASSERT(split_shexp_gate && split_shexp_gate->splits[id]);
                auto gate = llm_build_lora_mm(lctx, ctx, split_shexp_gate->splits[id], cur);
                // one op for every ubatch size: sigmoid + mul for a window only gave a T-token window and T single decodes different
                // state (test-spec-window-parity). Row S69-FORK-SHEXP-GATE-BATCH-VARIANT-SIGMOID-BRANCH-2026-11-05-01
                shared_out = ggml_fused_mul_unary(ctx, gate, shared_out, GGML_UNARY_OP_SIGMOID);
                cb(shared_out, "ffn_shexp_gated", il_cb);
            }

            cur = ggml_add(ctx, routed_out, shared_out);
            cb(cur, "ffn_out", il_cb);
        } else {
            cur = routed_out;
        }
        if (add_extra && add_extra->op == GGML_OP_REDUCE && add_extra->op_params[3] == 1 && add_extra->src[id]) {
            cur = ggml_add(ctx, cur, add_extra->src[id]);
            cb(cur, "ffn_with_extra", il_cb);
        }
        if (cur->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
            cur = ggml_cast(ctx, cur, lctx.cparams.reduce_type);
            cb(cur, "ffn_out_f16", il_cb);
        }
        ggml_build_forward_expand(graph, cur);
        results[id] = cur;
        last_id = id;
        down_bias_added = true;
    }
    GGML_ASSERT(last_id >= 0);
    if (add_input) {
        results[last_id] = ggml_add(ctx, results[last_id], input);
        cb(results[last_id], "ffn_inp_added", il);
    }
    if (add_extra && !(add_extra->op == GGML_OP_REDUCE && add_extra->op_params[3] == 1)) {
        results[last_id] = ggml_add(ctx, results[last_id], add_extra);
        cb(results[last_id], "ffn_with_inp", il);
    }
    if (add_extra && add_extra->op == GGML_OP_REDUCE && add_extra->op_params[3] == 1) {
        for (int id = 0; id < n_device; ++id) {
            if (!results[id]) results[id] = add_extra->src[id];
        }
    }

    auto cur = ggml_reduce(ctx, results.data(), n_device, GGML_OP_ADD);
    cb(cur, "moe_ffn_combined", il);
    ggml_build_forward_expand(graph, cur);

    return cur;
}

// Prints the XAttention selection totals at process exit when the op ran at least once (silent otherwise).
static justician_kv::xattn_exit_report g_xattn_exit_report;

static ggml_tensor * llm_build_kqv(
        struct ggml_context * ctx,
       struct llama_context & lctx,
       const llama_kv_cache_fork & kv,
         struct ggml_cgraph * graph,
         struct ggml_tensor * wo,
         struct ggml_tensor * wo_b,
         struct ggml_tensor * q_cur,
         struct ggml_tensor * kq_mask,
                    int32_t   n_tokens,
                    int32_t   n_kv,
                    float     kq_scale,
         const llm_build_cb & cb,
                    int       il,
                ggml_tensor * sinks = nullptr, int n_swa = 0, int kv_il = -1,
                ggml_tensor ** k_cache_view = nullptr, ggml_tensor ** v_cache_view = nullptr,
                    int32_t kv_view_offset = 0,
                ggml_tensor * k_fresh = nullptr, ggml_tensor * v_fresh = nullptr, int32_t fresh_head = -1) {
    const llama_model   & model   = lctx.model;
    const llama_hparams & hparams = lctx.model.hparams;
    const llama_cparams & cparams = lctx.cparams;

    const int     kv_layer      = kv_il >= 0 ? kv_il : il;
    const int64_t n_ctx         = kv.size;
    const int32_t n_cache_rows  = kv.rows(kv_layer);
    const int64_t n_head        = hparams.n_head(il);
    const int64_t n_head_kv     = hparams.n_head_kv(il);
    const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
    //const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_head_v = hparams.n_embd_head_v(il);
    const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(il);

    struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
    cb(q, "q", il);

    auto k_cache = kv_il >= 0 ? kv.k_l[kv_layer]
                 : lctx.model.hparams.has_kv(il) ? kv.k_l[il]
                 : lctx.model.hparams.is_swa_impl[il] ? kv.k_l[hparams.n_layer_kv_from_start-2] : kv.k_l[hparams.n_layer_kv_from_start-1];
    auto v_cache = kv_il >= 0 ? kv.v_l[kv_layer]
                 : lctx.model.hparams.has_kv(il) ? kv.v_l[il]
                 : lctx.model.hparams.is_swa_impl[il] ? kv.v_l[hparams.n_layer_kv_from_start-2] : kv.v_l[hparams.n_layer_kv_from_start-1];

    GGML_ASSERT(k_cache != nullptr && "k_cache is null in llm_build_kqv");
    GGML_ASSERT(v_cache != nullptr && "v_cache is null in llm_build_kqv");

    // S69 #66 KVFlash: pool-sized read view when the paged pool is active (gated;
    // default kvflash_pool_tokens==0 keeps n_kv => byte-identical).
    const int32_t n_kv_eff = (lctx.kvflash_pager && cparams.kvflash_pool_tokens > 0)
                           ? (int32_t) cparams.kvflash_pool_tokens : n_kv;

    struct ggml_tensor * k = k_cache_view ? *k_cache_view : nullptr;
    if (!k) {
        k = ggml_view_3d(ctx, k_cache,
                    n_embd_head_k, n_kv_eff, n_head_kv,
                    ggml_row_size(k_cache->type, n_embd_head_k)*n_head_kv, //n_embd_k_gqa),
                    ggml_row_size(k_cache->type, n_embd_head_k),
                    (size_t) kv_view_offset*ggml_row_size(k_cache->type, n_embd_head_k)*n_head_kv);
        if (k_cache_view) {
            *k_cache_view = k;
        }
        cb(k, "k", il);
    }

#ifdef GGML_USE_VULKAN
    constexpr bool use_f32_precision = true;
#else
    constexpr bool use_f32_precision = false;
#endif

    bool should_use_f32_precision = use_f32_precision
                                  || model.arch == LLM_ARCH_PHI2
                                  || model.arch == LLM_ARCH_PHI3
                                  || model.arch == LLM_ARCH_GPTNEOX
                                  || model.arch == LLM_ARCH_QWEN2
                                  || model.arch == LLM_ARCH_COHERE2
                                  || model.arch == LLM_ARCH_COHERE2_MOE
                                  || model.arch == LLM_ARCH_COMMAND_R
                                  || model.arch == LLM_ARCH_GLM4
                                  || model.arch == LLM_ARCH_GLM4_MOE
                                  || model.arch == LLM_ARCH_LAGUNA
                                  || model.arch == LLM_ARCH_MIMO2
                                  || model.arch == LLM_ARCH_K2_HORIZON;
                               // || (model.arch == LLM_ARCH_DEEPSEEK2 && q->ne[1] <= 8);

    struct ggml_tensor * cur;

    if (cparams.flash_attn) {
        GGML_UNUSED(model);
        GGML_UNUSED(n_ctx);

        // split cached v into n_head heads (not transposed)
        struct ggml_tensor * v = v_cache_view ? *v_cache_view : nullptr;
        if (!v) {
            v = ggml_view_3d(ctx, v_cache,
                        n_embd_head_v, n_kv_eff, n_head_kv,   // S69 #66 KVFlash pool-sized (gated)
                        ggml_row_size(v_cache->type, n_embd_v_gqa),
                        ggml_row_size(v_cache->type, n_embd_head_v),
                        (size_t) kv_view_offset*ggml_row_size(v_cache->type, n_embd_v_gqa));
            if (v_cache_view) {
                *v_cache_view = v;
            }
            cb(v, "v", il);
        }

        // S69 #220 offload-attention, FLASH-ATTENTION path. flash_attn_ext never materialises
        // the scores, so the only lever is the mask it consumes: select from keys+query and set
        // unselected pages to -INFINITY before handing it over. The mask is SHARED across heads,
        // so a page survives if ANY head wants it -- more conservative than the non-FA path, and
        // deliberately so, since one 2D mask cannot express per-head selection.
        struct ggml_tensor * kq_mask_eff = kq_mask;
        if (cparams.offload_attn && kq_mask) {
            static ik_offload::OffloadConfig oa_fa_cfg;
            oa_fa_cfg.page_size   = cparams.offload_attn_page;
            oa_fa_cfg.page_budget = cparams.offload_attn_budget;
            oa_fa_cfg.sink_pages  = cparams.offload_attn_sink;
            oa_fa_cfg.recent_pages = cparams.offload_attn_recent;
            oa_fa_cfg.min_kept_tokens = cparams.offload_attn_min_tokens;
            oa_fa_cfg.refresh_interval = cparams.offload_attn_refresh > 1 ? cparams.offload_attn_refresh : 1;
            oa_fa_cfg.scan_mode        = cparams.offload_attn_scan;
            if (cparams.offload_attn_head_reduce >= 0) oa_fa_cfg.head_reduce = cparams.offload_attn_head_reduce;   // --offload-attn-head-reduce; env GGML_OFFLOAD_ATTN_HEAD_REDUCE still overrides in the op
            oa_fa_cfg.selector    = (cparams.offload_attn_selector == 1) ? ik_offload::Selector::SHADOWKV
                                  : (cparams.offload_attn_selector == 2) ? ik_offload::Selector::RETATTN
                                                                         : ik_offload::Selector::QUEST;
            // DO NOT SPLICE AN OP THAT WOULD MASK NOTHING. When the token floor (or a large
            // budget) keeps every page, the mask is the identity and the node is pure cost:
            // MEASURED 30.8 tok/s against a 38.5 baseline with the op spliced and short-
            // circuiting INTERNALLY, because it still executes and copies the entire mask.
            // Deciding here removes the node instead. n_kv is known at build time from the
            // mask's own shape, and pages_kept() is the same function the op calls, so the two
            // cannot disagree about what would have been kept. When this test fails kq_mask_eff
            // stays kq_mask -- ordinary attention, the same fail-open direction the ops take.
            int64_t oa_np = 0;
            const int32_t oa_keep = ik_offload::pages_kept(oa_fa_cfg, kq_mask->ne[0], &oa_np);
            // DENSE LAYERS (--offload-attn-dense-layers N): the first N layers are never masked. HiP keeps layers 0-3 dense by
            // default (hip_config.py:366-372 dense_layers; hip_memory_pool.py:91) because early layers attend like a sliding
            // window (arXiv 2502.08910 App. D) and lose the most from page dropping. Decided here, at build time, from `il`:
            // the op itself has no layer index.
            ik_offload::DenseLayers oa_dense_set = ik_offload::dense_layers_from_cparams(
                    cparams.offload_attn_dense_layers, cparams.offload_attn_dense_listed, cparams.offload_attn_dense_mask);
            const bool oa_dense_layer = oa_dense_set.contains(il);
            const bool oa_masks = oa_keep < (int32_t) oa_np && !oa_dense_layer;
            if (oa_masks) {
                kq_mask_eff = ggml_map_custom3(ctx, kq_mask, k, q,
                                               ik_offload::mask_from_kq_op, GGML_N_TASKS_MAX,
                                               &oa_fa_cfg);
                cb(kq_mask_eff, "kq_mask_offload", il);
            }
            // ENGAGEMENT TRACE, always on, once per graph build. Without this the only evidence
            // that offload-attention ran is a throughput change, and "flag parses but the splice
            // never happens" produces the same clean output as "it ran" -- the exact shape that
            // let SnapKV/H2O be called wired while the serving launch passed none of their flags.
            // It now states whether the op was actually SPLICED, since "enabled and skipped as a
            // no-op" and "enabled and masking" are different facts that used to print the same.
            // Logged on the first layer only: per-layer would be n_layer lines per graph.
            if (il == 0) {
                LLAMA_LOG_INFO("%s: offload-attn ENGAGED (flash-attn path): page=%d budget=%.3f selector=%d sink=%d recent=%d dense_layers=%s head_reduce=%d min_tokens=%d keep=%d/%d %s\n",
                        __func__, (int) cparams.offload_attn_page,
                        (double) cparams.offload_attn_budget,
                        (int) cparams.offload_attn_selector,
                        (int) cparams.offload_attn_sink,
                        (int) cparams.offload_attn_recent,
                        oa_dense_set.describe((int) hparams.n_layer).c_str(),
                        (int) oa_fa_cfg.head_reduce,
                        (int) cparams.offload_attn_min_tokens,
                        (int) oa_keep, (int) oa_np,
                        oa_masks ? "SPLICED" : oa_dense_layer ? "NOT SPLICED on layer 0 (dense layer)" : "NOT SPLICED (identity: every page kept)");
            }
        }

        // XAttention prefill block selection (row S69-openvino-PICK-XATTENTION-PREFILL-BLOCK-SELECT-2026-09-29-01, src/llama-kv-xattn-op.h).
        // Spliced only where it can act: a prefill ubatch of whole selection blocks over at least two key blocks, threshold strictly below 1
        // (1.0 keeps every causal block, so the op would be pure cost). The op itself re-checks that every row is a single-sequence causal
        // prefix and copies the mask through otherwise; the CPU FA kernel then skips the all -inf tiles it wrote.
        {
            const justician_kv::xattn_op_cfg & xc = lctx.xattn_op;
            const bool xattn_splice = kq_mask_eff && xc.threshold > 0.0f && xc.threshold < 1.0f && xc.block > 0 &&
                                      n_tokens >= xc.block && n_tokens % xc.block == 0 && kq_mask_eff->ne[0] >= 2 * (int64_t) xc.block;
            if (xattn_splice) {
                kq_mask_eff = ggml_map_custom3(ctx, kq_mask_eff, k, q, justician_kv::xattn_mask_op, GGML_N_TASKS_MAX,
                                               (void *) &lctx.xattn_op);
                cb(kq_mask_eff, "kq_mask_xattn", il);
            }
            if (il == 0 && xc.threshold > 0.0f) {
                LLAMA_LOG_INFO("%s: xattn (flash-attn path): threshold=%.3f stride=%d block=%d n_tokens=%d n_kv=%lld %s\n", __func__,
                        (double) xc.threshold, (int) xc.stride, (int) xc.block, (int) n_tokens,
                        (long long) (kq_mask ? kq_mask->ne[0] : 0),
                        xattn_splice ? "SPLICED" : "NOT SPLICED (decode / partial block / threshold >= 1 / short KV)");
            }
        }

        // SnapKV/H2O attention mass under flash attention (fa_obs; row S69-OV-PICK-FA-OBSERVATION-WINDOW-ATTN-MASS-UNDER-FLASH-2026-10-06-01).
        // The FA kernel never materialises the softmax, so snapkv_eval_callback had nothing to read and common.cpp used to force -fa off
        // whenever --snapkv-cap / --h2o-heavy was set. A side branch below recomputes the post-softmax scores the way the non-FA path does
        // (mul_mat(k, q) -> soft_max_ext / softcap_max with the mask FA consumes, the same scale, ALiBi bias and sinks) and names the result
        // "kq_soft_max_ext" so the existing callback sums it into lctx.snapkv_obs. It is a leaf (nothing consumes it), built only for a
        // multi-token ubatch while the prune is armed (snapkv_active is decided before the graph is built), so every other graph is unchanged.
        // Not built (logged on layer 0) for a K cache the graph rotates (TurboQuant: Q is rotated into K's basis) or for a layer whose sliding
        // window the FA kernel applies itself (n_swa > 0): there the mask alone does not describe what FA attends to. The prune warns when
        // no layer contributed. Spec read: OpenVINO executor_pa.cpp:827-860,1064-1075,2008-2026 (Apache-2.0); own code, no code copied.
        ggml_tensor * q_fa_obs = q;
        // TurboQuant KV in host memory: the CPU backend has no FA wrapper that rotates, so the graph does (llama-turbo-graph.h; no-op for any other cache)
        q = llama_turbo_rotate_q(ctx, q, k_cache);
        if (lctx.snapkv_active && n_tokens > 1 && kq_mask_eff != nullptr) {
            const bool rotated = q != q_fa_obs;
            const bool fa_swa  = n_swa > 0 && !kv.cells_disordered;
            if (!rotated && !fa_swa) {
                ggml_tensor * kq_obs = ggml_mul_mat(ctx, k, q);
                ggml_mul_mat_set_prec(kq_obs, GGML_PREC_F32);
                if (hparams.attn_soft_cap) {
                    kq_obs = ggml_softcap_max(ctx, kq_obs, kq_mask_eff, kq_scale, hparams.f_max_alibi_bias,
                            1.0f / hparams.f_attn_logit_softcapping, hparams.f_attn_logit_softcapping);
                } else {
                    kq_obs = ggml_soft_max_ext(ctx, kq_obs, kq_mask_eff, kq_scale, hparams.f_max_alibi_bias);
                    ggml_soft_max_add_sinks(kq_obs, sinks);
                }
                cb(kq_obs, "kq_soft_max_ext", il);
                ggml_build_forward_expand(graph, kq_obs);
            }
            if (il == 0) {
                LLAMA_LOG_INFO("%s: snapkv/h2o attention mass (flash-attn path): %s\n", __func__,
                        rotated ? "not captured (rotated K cache)" : fa_swa ? "not captured on SWA layers" : "side branch SPLICED");
            }
        }

        // Fresh-chunk attention over a quantized KV cache (--kv-fresh-chunk; row
        // S69-OV-PICK-FRESH-CHUNK-FP16-KV-OVER-QUANTISED-CACHE-2026-10-06-01). The current ubatch's K/V were just written to the
        // cache in its quantized type, and FA then read them back: every query attended its own chunk through quantization error
        // it never needed, since the unquantized Kcur/Vcur are still in the graph. Here the cache view is dequantized (past cells
        // keep exactly their cached values, up to f16 rounding), the current ubatch's cells [fresh_head, fresh_head + n_tokens)
        // are overwritten with Kcur/Vcur, and FA runs on that f16 copy; the store into the cache is unchanged, so later ubatches
        // still see the quantized values. Mode 2 is the A/B control: the same dequantize -> f16 path WITHOUT the overwrite, so
        // whatever the f16 path changes on its own (FA no longer quantizes Q for a quantized K) is measured separately.
        // The caller (llm_build_kv) passes k_fresh/v_fresh only where the ubatch's cells are the contiguous block it stored at
        // fresh_head and graph reuse cannot move that head (can_reuse_graph refuses multi-token ubatches in this mode).
        // RESEARCH-DONE: OpenVINO executor_pa.cpp (Apache-2.0) attends the current chunk from the un-compressed K/V; spec read, own code.
        if (k_fresh && v_fresh && fresh_head >= 0 && q == q_fa_obs && cparams.kv_fresh_chunk > 0 &&
            (ggml_is_quantized(k->type) || ggml_is_quantized(v->type)) &&
            (int64_t) fresh_head + n_tokens <= k->ne[1] && (int64_t) fresh_head + n_tokens <= v->ne[1]) {
            auto fresh_view = [&](ggml_tensor * view, ggml_tensor * cur_kv, int64_t d, const char * name) -> ggml_tensor * {
                ggml_tensor * deq = ggml_cast(ctx, view, GGML_TYPE_F32);   // contiguous [d, n_kv, n_head_kv]
                if ((uint64_t) ggml_nbytes(deq) > (uint64_t) INT32_MAX) {
                    return nullptr;                                         // ggml_set carries strides/offset as int32
                }
                if (cparams.kv_fresh_chunk == 1) {
                    ggml_tensor * c = ggml_is_contiguous(cur_kv) ? cur_kv : ggml_cont(ctx, cur_kv);
                    if (c->type != GGML_TYPE_F32) {
                        c = ggml_cast(ctx, c, GGML_TYPE_F32);
                    }
                    c = ggml_reshape_3d(ctx, c, d, n_head_kv, n_tokens);
                    c = ggml_permute(ctx, c, 0, 2, 1, 3);                   // [d, n_tokens, n_head_kv], rows stay contiguous
                    deq = ggml_set_inplace(ctx, deq, c, deq->nb[1], deq->nb[2], deq->nb[3], (size_t) fresh_head * deq->nb[1]);
                }
                ggml_tensor * out = ggml_cast(ctx, deq, GGML_TYPE_F16);
                cb(out, name, il);
                return out;
            };
            ggml_tensor * k_f = fresh_view(k, k_fresh, n_embd_head_k, "k_fresh_chunk");
            ggml_tensor * v_f = k_f ? fresh_view(v, v_fresh, n_embd_head_v, "v_fresh_chunk") : nullptr;
            if (k_f && v_f) {
                k = k_f;
                v = v_f;
            }
            if (il == 0) {
                LLAMA_LOG_INFO("%s: kv fresh-chunk mode %d: %s (head=%d n_tokens=%d n_kv=%lld)\n", __func__, cparams.kv_fresh_chunk,
                        (k_f && v_f) ? (cparams.kv_fresh_chunk == 1 ? "SPLICED" : "CONTROL (dequantized, no overwrite)") : "NOT SPLICED (view > 2 GiB)",
                        (int) fresh_head, (int) n_tokens, (long long) k->ne[1]);
            }
        }
        cur = ggml_flash_attn_ext(ctx, q, k, v, kq_mask_eff, kq_scale, hparams.f_max_alibi_bias,
                hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        cb(cur, "fa", il);
        ggml_flash_attn_ext_add_sinks(cur, sinks);
        if (n_swa > 0 && !kv.cells_disordered && cparams.n_seq_max == 1) {
            ((int32_t *)cur->op_params)[4] = n_swa;
        }

        // Some models produced NaNs/gibberish when FA is computed with f16 precision on CUDA
        // For DeepSeek-2, it is perfectly fine with fp16 for PP, but I get gibberish when uding fp16 for TG.
        // Not sure if it is really a matter of insufficient precision, or I have made a mistake in the fattn-vec-f16 kernel.
        if (should_use_f32_precision) {
            ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        }
        //ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = llama_turbo_unrotate_out(ctx, cur, v_cache);

        if (cparams.v_cache_hadamard) {
            if (int block_size = lctx.model.hadamard_size_v(il); block_size > 0) {
                cur = ggml_hadamard(ctx, cur, block_size);
                cb(cur, "fa_h", il);
            }
        }
        cur = ggml_reshape_2d(ctx, cur, n_embd_head_v*n_head, n_tokens);
    } else {

            // split cached v into n_head heads
        struct ggml_tensor * v = v_cache_view ? *v_cache_view : nullptr;
        if (!v) {
            if (kv.v_trans) {
                v = ggml_view_3d(ctx, v_cache,
                        n_kv, n_embd_head_v, n_head_kv,
                        ggml_element_size(v_cache)*n_cache_rows,
                        ggml_element_size(v_cache)*n_cache_rows*n_embd_head_v,
                        (size_t) kv_view_offset*ggml_element_size(v_cache));
            } else {
                v = ggml_view_3d(ctx, v_cache,
                        n_embd_head_v, n_kv_eff, n_head_kv,   // S69 #66 KVFlash pool-sized (gated)
                        ggml_row_size(v_cache->type, n_embd_v_gqa),
                        ggml_row_size(v_cache->type, n_embd_head_v),
                        (size_t) kv_view_offset*ggml_row_size(v_cache->type, n_embd_v_gqa));
                v = ggml_cont(ctx, ggml_transpose(ctx, v));
            }
            if (v_cache_view) {
                *v_cache_view = v;
            }
            cb(v, "v", il);
        }

        auto kq_size = k->ne[1]*q->ne[1]*q->ne[2]*sizeof(float)/(1024*1024);
        if (cparams.attn_max_batch == 0 || cparams.attn_max_batch >= kq_size || k->ne[2] != q->ne[2] || v->ne[2] != q->ne[2] || sinks) {
            struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            cb(kq, "kq", il);
            // S69 #220 offload-attention, NON-FA path. Splice the masking op between the scores
            // and the softmax so unselected KV pages get -INFINITY and therefore exactly zero
            // weight. This is the only place the selection CAN happen: the builder has no KV
            // contents, so the decision has to be made inside an operator that runs with real data.
            // The op fails open and counts pass-throughs, so a config it cannot handle leaves the
            // model computing exactly what it did before, visibly rather than silently.
            if (cparams.offload_attn) {
                static ik_offload::OffloadConfig oa_cfg;
                oa_cfg.page_size   = cparams.offload_attn_page;
                oa_cfg.page_budget = cparams.offload_attn_budget;
                oa_cfg.sink_pages  = cparams.offload_attn_sink;
                oa_cfg.recent_pages = cparams.offload_attn_recent;
                oa_cfg.min_kept_tokens = cparams.offload_attn_min_tokens;
                oa_cfg.selector    = (cparams.offload_attn_selector == 1) ? ik_offload::Selector::SHADOWKV
                                   : (cparams.offload_attn_selector == 2) ? ik_offload::Selector::RETATTN
                                                                          : ik_offload::Selector::QUEST;
                // Do not splice an op that would mask nothing -- see the matching note on the
                // flash-attention path above. Same pages_kept() call, so the two paths cannot
                // reach different conclusions about whether masking would change anything.
                int64_t oa_np = 0;
                const int32_t oa_keep = ik_offload::pages_kept(oa_cfg, kq->ne[0], &oa_np);
                const ik_offload::DenseLayers oa_dense_set = ik_offload::dense_layers_from_cparams(
                        cparams.offload_attn_dense_layers, cparams.offload_attn_dense_listed, cparams.offload_attn_dense_mask);
                const bool oa_dense_layer = oa_dense_set.contains(il);   // see the FA path above
                if (cparams.offload_attn_head_reduce >= 0) oa_cfg.head_reduce = cparams.offload_attn_head_reduce;
                const bool oa_masks = oa_keep < (int32_t) oa_np && !oa_dense_layer;
                if (oa_masks) {
                    // With the attention mask as a third input the op scores only keys this query may attend and places the recent
                    // window at the query's real position (kq is RAW here; kq_mask is applied later inside the softmax). Same tensor
                    // the softmax below consumes. Without a usable mask, the two-input op (old behaviour).
                    if (kq_mask && kq_mask->ne[0] == kq->ne[0]) {
                        kq = ggml_map_custom3(ctx, kq, k, kq_mask, ik_offload::mask_op_kqmask, GGML_N_TASKS_MAX, &oa_cfg);
                    } else {
                        kq = ggml_map_custom2(ctx, kq, k, ik_offload::mask_op, GGML_N_TASKS_MAX, &oa_cfg);
                    }
                    cb(kq, "kq_offload_masked", il);
                }
                // ENGAGEMENT TRACE, always on, once per graph build -- see the matching note on
                // the flash-attention path above. The two lines name WHICH path spliced, because
                // only one of them runs for a given -fa setting and "offload-attn engaged" alone
                // would not say which.
                if (il == 0) {
                    LLAMA_LOG_INFO("%s: offload-attn ENGAGED (non-FA path): page=%d budget=%.3f selector=%d sink=%d recent=%d dense_layers=%s head_reduce=%d min_tokens=%d keep=%d/%d %s\n",
                            __func__, (int) cparams.offload_attn_page,
                            (double) cparams.offload_attn_budget,
                            (int) cparams.offload_attn_selector,
                            (int) cparams.offload_attn_sink,
                            (int) cparams.offload_attn_recent,
                            oa_dense_set.describe((int) hparams.n_layer).c_str(),
                            (int) oa_cfg.head_reduce,
                            (int) cparams.offload_attn_min_tokens,
                            (int) oa_keep, (int) oa_np,
                            oa_masks ? "SPLICED" : oa_dense_layer ? "NOT SPLICED on layer 0 (dense layer)" : "NOT SPLICED (identity: every page kept)");
                }
            }

            //ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

            if (should_use_f32_precision) {
                // for this arch, we need to perform the KQ multiplication with F32 precision, otherwise we get NaNs
                // ref: https://github.com/ggerganov/llama.cpp/pull/4490#issuecomment-1859055847
                ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            }

            if (model.arch == LLM_ARCH_GROK) {
                // need to do the following:
                // multiply by attn_output_multiplier
                // and then :
                // kq = 30 * tanh(kq / 30)
                // before the softmax below

                //try from phi2
                //ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

                //kq = ggml_tanh(ctx, ggml_scale(ctx, kq, 0.08838834764831845f/30.0f));
                //kq = ggml_scale(ctx, kq, 30);

                kq = ggml_softcap(ctx, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping, hparams.f_attn_logit_softcapping);
            }

            if (hparams.attn_soft_cap) {
                //kq = ggml_softcap(ctx, kq, 1.0f / hparams.f_attn_logit_softcapping, hparams.f_attn_logit_softcapping);
                kq = ggml_softcap_max(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                        1.0f / hparams.f_attn_logit_softcapping, hparams.f_attn_logit_softcapping);
            } else {
                kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
                ggml_soft_max_add_sinks(kq, sinks);
            }
            cb(kq, "kq_soft_max_ext", il);

            GGML_ASSERT(kv.size == n_ctx);

            struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
            cb(kqv, "kqv", il);

            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);

            cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_v*n_head, n_tokens);
            cb(cur, "kqv_merged_cont", il);
        }
        else {
            // For now we will not support this option if k->ne[2] != q->ne[2] || v->ne[2] != q->ne[2];
            GGML_ASSERT(k->ne[2] == v->ne[2] && k->ne[2] == q->ne[2]);
            int n_step = (kq_size + cparams.attn_max_batch - 1)/cparams.attn_max_batch;
            n_step = std::min(n_step, int(k->ne[2]));
            int n_per_step = (q->ne[2] + n_step - 1)/n_step;
            auto r2k = q->ne[2] / k->ne[2];
            auto r2v = q->ne[2] / v->ne[2];
            n_step = q->ne[2];
            n_per_step = 1;
            ggml_tensor * kqv = nullptr;
            for (int i12 = 0; i12 < q->ne[2]; i12 += n_per_step) {
                int this_ne12 = i12 + n_per_step <= q->ne[2] ? n_per_step : q->ne[2] - i12;
                int i02 = i12/r2k;
                auto k_i = ggml_view_3d(ctx, k, k->ne[0], k->ne[1], this_ne12, k->nb[1], k->nb[2], k->nb[2]*i02);
                auto q_i = ggml_view_3d(ctx, q, q->ne[0], q->ne[1], this_ne12, q->nb[1], q->nb[2], q->nb[2]*i12);
                auto kq_i = ggml_mul_mat(ctx, k_i, q_i);
                if (model.arch == LLM_ARCH_PHI2 || model.arch == LLM_ARCH_PHI3 || model.arch == LLM_ARCH_GPTNEOX || model.arch == LLM_ARCH_QWEN2 ||
                    model.arch == LLM_ARCH_COHERE2 || model.arch == LLM_ARCH_COHERE2_MOE || model.arch == LLM_ARCH_COMMAND_R || model.arch == LLM_ARCH_GLM4 || model.arch == LLM_ARCH_GLM4_MOE) {
                    ggml_mul_mat_set_prec(kq_i, GGML_PREC_F32);
                }
                if (model.arch == LLM_ARCH_GROK) {
                    kq_i = ggml_softcap(ctx, kq_i, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping, hparams.f_attn_logit_softcapping);
                }
                if (hparams.attn_soft_cap) {
                    kq_i = ggml_softcap_max(ctx, kq_i, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                            1.0f / hparams.f_attn_logit_softcapping, hparams.f_attn_logit_softcapping);
                } else {
                    kq_i = ggml_soft_max_ext(ctx, kq_i, kq_mask, kq_scale, hparams.f_max_alibi_bias);
                }
                i02 = i12 / r2v;
                auto v_i = ggml_view_3d(ctx, v, v->ne[0], v->ne[1], this_ne12, v->nb[1], v->nb[2], v->nb[2]*i02);
                auto kqv_i = ggml_mul_mat(ctx, v_i, kq_i);
                if (i12 == 0) {
                    kqv = kqv_i;
                } else {
                    kqv = ggml_concat(ctx, kqv, kqv_i, 2);
                }
            }
            ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);
            cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_v*n_head, n_tokens);
            cb(cur, "kqv_merged_cont", il);
        }
    }

    ggml_build_forward_expand(graph, cur);

    if (wo) {
        cur = llm_build_context::llm_build_lora_mm(lctx, ctx, wo, cur);
        if (lctx.model.arch == LLM_ARCH_GLM4 || lctx.model.arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (wo_b) {
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_build_context::llm_build_kv(
        ggml_context * ctx,
       llama_context & lctx,
       const llama_kv_cache_fork & kv,
         ggml_cgraph * graph,
         ggml_tensor * wo,
         ggml_tensor * wo_b,
         ggml_tensor * k_cur,
         ggml_tensor * v_cur,
         ggml_tensor * q_cur,
         ggml_tensor * kq_mask,
                    int32_t   n_tokens,
                    int32_t   kv_head,
                    int32_t   n_kv,
                    float     kq_scale,
         const llm_build_cb & cb, int il, ggml_tensor * sinks, int n_swa, int kv_il,
         ggml_tensor ** k_cache_view, ggml_tensor ** v_cache_view,
                    int32_t   swa_head) {
    const llama_hparams & hparams = lctx.model.hparams;
    const llama_cparams & cparams = lctx.cparams;

    if (cparams.k_cache_hadamard) {
        if (int block_size = lctx.model.hadamard_size_k(il); block_size > 0) {
            q_cur = ggml_hadamard(ctx, q_cur, block_size);
            if (k_cur) {
                k_cur = ggml_hadamard(ctx, k_cur, block_size);
                cb(k_cur, "Kcur_hadamard", il);
            }
            cb(q_cur, "Qcur_hadamard", il);
        }
    }
    if (cparams.v_cache_hadamard && v_cur) {
        if (int block_size = lctx.model.hadamard_size_v(il); block_size > 0) {
            v_cur = ggml_hadamard(ctx, v_cur, block_size);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(graph, q_cur);
    if (k_cur) {
        ggml_build_forward_expand(graph, k_cur);
    }
    if (v_cur) {
        ggml_build_forward_expand(graph, v_cur);
    }

    const bool compacted = kv.is_compacted(kv_il >= 0 ? kv_il : il);
    const bool use_swa_window = compacted && lctx.swa_window_view.active;
    const int32_t store_head = compacted ? swa_head : kv_head;
    const int32_t n_kv_view = use_swa_window ? (int32_t) lctx.swa_window_view.w_view : n_kv;
    const int32_t kv_view_offset = use_swa_window ? (int32_t) lctx.swa_window_view.win_off : 0;

    if (k_cur || v_cur) {
        llm_build_kv_store(lctx, ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, store_head, cb, il);
    }

    // Fresh-chunk (--kv-fresh-chunk, see llm_build_kqv): only where this ubatch's K/V are the contiguous block just stored at
    // store_head of an uncompacted, ordered cache read through an unshared, unpaged view starting at cell 0.
    const bool fresh_ok = cparams.kv_fresh_chunk > 0 && cparams.flash_attn && k_cur && v_cur && n_tokens > 1 && !compacted &&
                          kv_view_offset == 0 && !kv.cells_disordered && k_cache_view == nullptr && v_cache_view == nullptr &&
                          !(lctx.kvflash_pager && cparams.kvflash_pool_tokens > 0) && store_head >= 0 && (kv_il < 0 || kv_il == il) &&
                          store_head + n_tokens <= n_kv_view;
    auto cur = llm_build_kqv(ctx, lctx, kv, graph, wo, wo_b, q_cur, kq_mask, n_tokens, n_kv_view, kq_scale, cb, il, sinks, n_swa, kv_il,
            k_cache_view, v_cache_view, kv_view_offset,
            fresh_ok ? k_cur : nullptr, fresh_ok ? v_cur : nullptr, fresh_ok ? store_head : -1);
    cb(cur, "kqv_out", il);

    return cur;
}

ggml_tensor * llm_build_context::llm_build_inp_embd_enc() {
    const int64_t n_embd = hparams.n_embd;
    lctx.inp_embd_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_outputs_enc);
    ggml_set_input(lctx.inp_embd_enc);
    cb(lctx.inp_embd_enc, "embd_enc", -1);
    return lctx.inp_embd_enc;
}

ggml_tensor * llm_build_context::llm_build_inp_KQ_mask_cross() {
    lctx.inp_KQ_mask_cross = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_outputs_enc, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    ggml_set_input(lctx.inp_KQ_mask_cross);
    cb(lctx.inp_KQ_mask_cross, "KQ_mask_cross", -1);
    return lctx.inp_KQ_mask_cross;
}

std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> llm_build_context::llm_build_mul_mat_qkv(ggml_cgraph * gf, ggml_tensor * cur,
            ggml_tensor * wq, ggml_tensor * bq,
            ggml_tensor * wk, ggml_tensor * bk,
            ggml_tensor * wv, ggml_tensor * bv,
            float attention_scale, int il, bool add_graph_split) const {
    auto Qcur = llm_build_lora_mm(lctx, ctx0, wq, cur);
    cb(Qcur, "Qcur", il);
    if (add_graph_split) {
        Qcur->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
    }
    auto Kcur = llm_build_lora_mm(lctx, ctx0, wk, cur);
    cb(Kcur, "Kcur", il);
    auto Vcur = llm_build_lora_mm(lctx, ctx0, wv, cur);
    cb(Vcur, "Vcur", il);
    ggml_build_forward_expand(gf, Qcur);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, Vcur);

    if (attention_scale != 0) {
        Qcur = ggml_scale(ctx0, Qcur, attention_scale);
        cb(Qcur, "Qcur", il);
    }
    if (bq) {
        Qcur = ggml_add(ctx0, Qcur, bq);
        cb(Qcur, "Qcur", il);
        ggml_build_forward_expand(gf, Qcur);
    }
    if (bk) {
        Kcur = ggml_add(ctx0, Kcur, bk);
        cb(Kcur, "Kcur", il);
        ggml_build_forward_expand(gf, Kcur);
    }
    if (bv) {
        Vcur = ggml_add(ctx0, Vcur, bv);
        cb(Vcur, "Vcur", il);
        ggml_build_forward_expand(gf, Vcur);
    }
    return {Qcur, Kcur, Vcur};
}

std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*> llm_build_context::llm_build_mul_mat_qkv_gated(ggml_cgraph * gf, ggml_tensor * cur,
            ggml_tensor * wq, ggml_tensor * wk, ggml_tensor * wv, ggml_tensor * q_norm, ggml_tensor * k_norm, int il) const {
    auto Qaux = llm_build_lora_mm(lctx, ctx0, wq, cur);
    cb(Qaux, "Qaux", il);
    auto Kcur = llm_build_lora_mm(lctx, ctx0, wk, cur);
    cb(Kcur, "Kcur", il);
    auto Vcur = llm_build_lora_mm(lctx, ctx0, wv, cur);
    cb(Vcur, "Vcur", il);
    ggml_build_forward_expand(gf, Qaux);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, Vcur);
    auto row_size = ggml_row_size(Qaux->type, n_embd_head_k);
    // TODO: check why CUDA performance suffers so much if we don't make these two tensors contiguous
    auto Qcur = ggml_cont(ctx0, ggml_view_3d(ctx0, Qaux, n_embd_head_k, Qaux->ne[0]/(2*n_embd_head_k), n_tokens, 2*row_size, Qaux->nb[1], 0));
    cb(Qcur, "Qcur_cont", il);
    auto gate = ggml_cont_2d(ctx0, ggml_view_3d(ctx0, Qaux, n_embd_head_k, Qaux->ne[0]/(2*n_embd_head_k), n_tokens, 2*row_size, Qaux->nb[1], row_size), Qaux->ne[0]/2, n_tokens);
    cb(gate, "gate_cont", il);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, Kcur->ne[0]/n_embd_head_k, n_tokens);
    if (q_norm) {
        Qcur = llm_build_norm(ctx0, Qcur, hparams, q_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(Qcur, "Qcur_normed", il);
        ggml_build_forward_expand(gf, Qcur);
    }
    if (k_norm) {
        Kcur = llm_build_norm(ctx0, Kcur, hparams, k_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(Kcur, "Kcur_normed", il);
        ggml_build_forward_expand(gf, Kcur);
    }
    //gate = ggml_sigmoid(ctx0, gate);
    //gate = ggml_reshape_2d(ctx0, gate, gate->ne[0]*gate->ne[1], gate->ne[2]);
    //cb(gate, "gate", il);
    return {Qcur, Kcur, Vcur, gate};
}

std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> llm_build_context::llm_build_mul_mat_qkv(ggml_cgraph * gf, ggml_tensor * cur,
            ggml_tensor * wqkv, ggml_tensor * bqkv,
            ggml_tensor * wqk, ggml_tensor * bqk,
            ggml_tensor * wq, ggml_tensor * bq,
            ggml_tensor * wk, ggml_tensor * bk,
            ggml_tensor * wv, ggml_tensor * bv,
            ggml_tensor * q_norm, ggml_tensor * k_norm, float attention_scale, int il, bool add_graph_split) const {
    int n_head    = hparams.n_head(il);
    int n_head_kv = hparams.n_head_kv(il);
    const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
    if (wqkv) {
        auto qkv = llm_build_lora_mm(lctx, ctx0, wqkv, cur);
        if (add_graph_split) {
            qkv->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
        }
        cb(qkv, "qkv", il);
        if (bqkv) {
            qkv = ggml_add(ctx0, qkv, bqkv);
            cb(qkv, "qkv_b", il);
        }
        auto Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head,    n_tokens, n_embd_head_k*sizeof(float), qkv->nb[1], 0*sizeof(float)*(n_embd));
        auto Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_kv, n_tokens, n_embd_head_k*sizeof(float), qkv->nb[1], 1*sizeof(float)*Qcur->ne[0]*Qcur->ne[1]);
        auto Vcur = ggml_view_2d(ctx0, qkv, n_embd_gqa, n_tokens, qkv->nb[1], 1*sizeof(float)*(Qcur->ne[0]*Qcur->ne[1] + Kcur->ne[0]*Kcur->ne[1]));
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        if (q_norm) {
            Qcur = llm_build_norm(ctx0, Qcur, hparams, q_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur_normed", il);
            ggml_build_forward_expand(gf, Qcur);
        }
        if (k_norm) {
            Kcur = llm_build_norm(ctx0, Kcur, hparams, k_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur_normed", il);
            ggml_build_forward_expand(gf, Kcur);
        }

        return {Qcur, Kcur, Vcur};

        //ggml_build_forward_expand(gf, Qcur);
        //ggml_build_forward_expand(gf, Kcur);
        //ggml_build_forward_expand(gf, Vcur);
    }

    if (wqk) {
        auto qk = llm_build_lora_mm(lctx, ctx0, wqk, cur);
        if (add_graph_split) {
            qk->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
        }
        cb(qk, "qkv", il);
        if (bqk) {
            qk = ggml_add(ctx0, qk, bqk);
            cb(qk, "qkv_b", il);
        }
        auto Vcur = llm_build_lora_mm(lctx, ctx0, wv, cur);
        cb(Vcur, "Vcur", il);
        if (bv) {
            Vcur = ggml_add(ctx0, Vcur, bv);
            cb(Vcur, "Vcur", il);
        }
        ggml_build_forward_expand(gf, qk);
        ggml_build_forward_expand(gf, Vcur);
        auto Qcur = ggml_view_3d(ctx0, qk, n_embd_head_k, n_head,    n_tokens, n_embd_head_k*sizeof(float), qk->nb[1], 0*sizeof(float)*(n_embd));
        auto Kcur = ggml_view_3d(ctx0, qk, n_embd_head_k, n_head_kv, n_tokens, n_embd_head_k*sizeof(float), qk->nb[1], 1*sizeof(float)*Qcur->ne[0]*Qcur->ne[1]);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        if (q_norm) {
            Qcur = llm_build_norm(ctx0, Qcur, hparams, q_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur_normed", il);
            ggml_build_forward_expand(gf, Qcur);
        }
        if (k_norm) {
            Kcur = llm_build_norm(ctx0, Kcur, hparams, k_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur_normed", il);
            ggml_build_forward_expand(gf, Kcur);
        }

        return {Qcur, Kcur, Vcur};

    }

    auto [Q, K, V] = llm_build_mul_mat_qkv(gf, cur, wq, bq, wk, bk, wv, bv, attention_scale, il, add_graph_split);
    auto Qcur = ggml_reshape_3d(ctx0, Q, n_embd_head_k, Q->ne[0]/n_embd_head_k, n_tokens);
    // Command-R/R+ uses LayerNorm (not RMSNorm) for per-head Q/K normalization
    const auto qk_norm_type = (model.arch == LLM_ARCH_COMMAND_R) ? LLM_NORM : LLM_NORM_RMS;
    if (q_norm) {
        Qcur = llm_build_norm(ctx0, Qcur, hparams, q_norm, NULL, qk_norm_type, cb, il);
        cb(Qcur, "Qcur_normed", il);
    }

    auto Kcur = ggml_reshape_3d(ctx0, K, n_embd_head_k, K->ne[0]/n_embd_head_k, n_tokens);
    if (k_norm) {
        Kcur = llm_build_norm(ctx0, Kcur, hparams, k_norm, NULL, qk_norm_type, cb, il);
        cb(Kcur, "Kcur_normed", il);
    }
    auto Vcur = V;
    return {Qcur, Kcur, Vcur};
}

// The output projection. With llama_set_output_vocab_subset() active it multiplies by the partial weight (the rows of the output weight listed in the
// subset, same type, gathered once) instead of the whole vocabulary: CTranslate2's Dense::select_weights (src/layers/common.cc:313-335). Only on a real
// graph (the worst-case reservation keeps the full projection so the compute buffer is sized for it), only for the model's own output weight on a single
// device, and not when LoRA / an output bias would have to be applied to the same product.
static ggml_tensor * build_output_mm(llama_context & lctx, ggml_context * ctx, ggml_tensor * output, ggml_tensor * cur) {
    auto & sub = lctx.output_subset;
    if (sub.partial && sub.allowed && output == lctx.model.output && !llm_build_context::llm_lora_targets(lctx, output) && lctx.model.output_b == nullptr && !lctx.cparams.mtp) {
        sub.in_graph = true;
        return ggml_mul_mat(ctx, sub.partial, cur);
    }
    return llm_build_context::llm_build_lora_mm(lctx, ctx, output, cur);
}

ggml_tensor * llm_build_context::build_output(llama_context & lctx, ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * output, const llm_build_cb & cb) {
    // lm_head
    if (output->extra) {
        auto split_output = (ggml_split_tensor_t *)output->extra;
        std::vector<ggml_tensor *> o;
        o.reserve(split_output->n_device);
        for (int id = 0; id < split_output->n_device; ++id) {
            auto split = split_output->splits[id];
            if (!split) continue;
            o.push_back(llm_build_context::llm_build_lora_mm(lctx, ctx, split, cur));
            cb(o.back(), "output", id);
        }
        if (o.size() == 1) cur = o.front();
        cur = ggml_concat(ctx, o[0], o[1], 0);
        for (int id = 2; id < int(o.size()); ++id) {
            cur = ggml_concat(ctx, cur, o[id], 0);
        }
    } else {
        cur = build_output_mm(lctx, ctx, output, cur);
    }
    return cur;
}

ggml_tensor * llm_build_context::build_output(llama_context & lctx, ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * output, ggml_tensor * output_norm, const llm_build_cb & cb, bool add_normed_name) {
    // lm_head
    if (output->extra) {
        auto split_output = (ggml_split_tensor_t *)output->extra;
        auto split_output_norm = output_norm && output_norm->extra ? (ggml_split_tensor_t *)output_norm->extra : nullptr;
        std::vector<ggml_tensor *> o;
        o.reserve(split_output->n_device);
        ggml_tensor * last_norm = nullptr;
        for (int id = 0; id < split_output->n_device; ++id) {
            auto split = split_output->splits[id];
            if (!split) continue;
            if (output_norm) {
                auto the_norm = split_output_norm ? split_output_norm->splits[id] : output_norm;
                auto cur_normed = llm_build_context::llm_build_norm(ctx, cur, lctx.model.hparams, the_norm, NULL, LLM_NORM_RMS, cb, -1);
                last_norm = cur_normed;
                cb(cur_normed, "result_norm", 1000*(id+1));
                o.push_back(llm_build_context::llm_build_lora_mm(lctx, ctx, split, cur_normed));
            } else {
                o.push_back(llm_build_context::llm_build_lora_mm(lctx, ctx, split, cur));
            }
            cb(o.back(), "output", id);
            if (add_normed_name && last_norm) {
                cb(last_norm, "result_norm", -1);
            }
        }
        GGML_ASSERT(!o.empty());
        if (o.size() == 1) {
            cur = o.front();
        }
        else {
            cur = ggml_concat(ctx, o[0], o[1], 0);
            for (int id = 2; id < int(o.size()); ++id) {
                cur = ggml_concat(ctx, cur, o[id], 0);
            }
        }
    } else {
        int idx = lctx.model.default_layer_device[lctx.model.hparams.n_layer];
        int idx_out = ggml_backend_sched_get_backend_idx(lctx.sched, lctx.model.output->buffer);
        if (idx_out >= 0) idx = idx_out;
        const bool is_qwen_mtp = (lctx.model.arch == LLM_ARCH_QWEN35 ||
                                  lctx.model.arch == LLM_ARCH_QWEN35MOE ||
                                  lctx.model.arch == LLM_ARCH_GLM4_MOE) && lctx.cparams.mtp;
        if (cur->op == GGML_OP_REDUCE && cur->src[idx] && !is_qwen_mtp) {
            // avoid copy to main GPU
            cur->view_src = cur->src[idx];
        }
        if (output_norm) {
            cur = llm_build_context::llm_build_norm(ctx, cur, lctx.model.hparams, output_norm, NULL, LLM_NORM_RMS, cb, -1);
            if (add_normed_name) {
                cb(cur, "result_norm", -1);
            }
        }
        cur = build_output_mm(lctx, ctx, output, cur);
    }
    return cur;
}

ggml_cgraph * llm_build_context::llama_build_graph_defrag(llama_context & lctx, const std::vector<uint32_t> & ids) {
    llama_batch dummy;
    dummy.n_tokens = 0;

    llm_build_cb cb = [&](struct ggml_tensor * , const char * , int ) { };

    struct llm_build_context llm(lctx, dummy, cb, false, false);

    llm.init();

    struct ggml_cgraph * result = llm.build_defrag(ids);

    llm.free();

    return result;
}

ggml_cgraph * llm_build_context::llama_build_graph_k_shift(llama_context & lctx) {
    llama_batch dummy;
    dummy.n_tokens = 0;

    llm_build_cb cb = [&](struct ggml_tensor * , const char * , int ) { };

    struct llm_build_context llm(lctx, dummy, cb, false, false);

    llm.init();

    struct ggml_cgraph * result = llm.build_k_shift();

    llm.free();

    return result;
}

struct ggml_cgraph * llm_build_context::llama_build_graph_s_copy(llama_context & lctx) {
    llama_batch dummy;
    dummy.n_tokens = 0;

    llm_build_cb cb = [&](struct ggml_tensor * , const char * , int ) { };

    struct llm_build_context llm(lctx, dummy, cb, false, false);

    llm.init();

    struct ggml_cgraph * result = llm.build_s_copy();

    llm.free();

    return result;
}

struct ggml_cgraph * llm_build_context::llama_build_graph_dflash_kv_cache(llama_context & lctx) {
    llama_batch dummy;
    dummy.n_tokens = 0;

    llm_build_cb cb = [&](struct ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            int j = 0;
            for (; j < GGML_MAX_NAME - 1; ++j) {
                cur->name[j] = name[j];
                if (!name[j]) {
                    break;
                }
            }
            if (j < GGML_MAX_NAME - 3) {
                cur->name[j++] = '-';
                auto sil = std::to_string(il);
                for (int k = 0; k < (int) sil.size() && j < GGML_MAX_NAME - 1; ++k) {
                    cur->name[j++] = sil[k];
                }
            }
            cur->name[j] = 0;
        } else {
            ggml_set_name(cur, name);
        }
    };

    struct llm_build_context llm(lctx, dummy, cb, false, false, 0, false, &lctx.dflash.kv.cache_compute_meta);

    llm.init();

    struct ggml_cgraph * result = llm.build_dflash_kv_cache();

    llm.free();

    return result;
}

ggml_cgraph * llm_build_context::llama_build_graph(
         llama_context & lctx,
     const llama_batch & batch,
                  bool   worst_case,
                  int    n_outputs) {
    const auto & model = lctx.model;

#if IK_PRINT_TIMING
    auto tim1 = ggml_time_us();
#endif

    // this callback allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
    llm_build_cb cb = [&](struct ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            int j = 0;
            for (; j < GGML_MAX_NAME - 1; ++j) {
                cur->name[j] = name[j];
                if (!name[j]) break;
            }
            if (j < GGML_MAX_NAME - 3) {
                cur->name[j++] = '-';
                auto sil = std::to_string(il);
                for (int k = 0; k < (int)sil.size() && j < GGML_MAX_NAME - 1; ++k) {
                    cur->name[j++] = sil[k];
                }
            }
            cur->name[j] = 0;
            //ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!lctx.cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(lctx.sched, cur, lctx.backend_cpu);
            }
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = lctx.model.n_gpu_layers > (int)lctx.model.hparams.n_layer;
        if (batch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                for (auto * backend : lctx.backends) {
                    if (ggml_backend_supports_buft(backend, lctx.model.buft_layer[il].buft) &&
                        (ggml_backend_supports_op(backend, cur) || ggml_backend_offload_op(backend, cur))) {
                        ggml_backend_sched_set_tensor_backend(lctx.sched, cur, backend);
                        break;
                    }
                }
            }
        }
    };

    struct ggml_cgraph * result = NULL;

    const llama_vocab * vocab = &lctx.model.vocab; //llama_get_vocab(&lctx);
    llama_token bos = vocab->token_bos();
    llama_token eos = vocab->token_eos();
    bool is_warming_up = lctx.n_eval == 0 && (batch.n_tokens == 1 && (batch.token[0] == ((bos != -1) ? bos : eos)));
    lctx.output_subset.allowed  = !worst_case;
    lctx.output_subset.in_graph = false;
    struct llm_build_context llm(lctx, batch, cb, worst_case, is_warming_up, n_outputs);

    llm.init();

    switch (model.arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
            {
                result = llm.build_llama();
            } break;
        case LLM_ARCH_MUSE_GLIMMER:
            {
                result = llm.build_muse_glimmer();
            } break;
        case LLM_ARCH_DECI:
            {
                result = llm.build_deci();
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                result = llm.build_baichuan();
            } break;
        case LLM_ARCH_FALCON:
            {
                result = llm.build_falcon();
            } break;
        case LLM_ARCH_GROK:
            {
                result = llm.build_grok();
            } break;
        case LLM_ARCH_STARCODER:
            {
                result = llm.build_starcoder();
            } break;
        case LLM_ARCH_REFACT:
            {
                result = llm.build_refact();
            } break;
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_NOMIC_BERT:
            {
                result = llm.build_bert();
            } break;
        case LLM_ARCH_BLOOM:
            {
                result = llm.build_bloom();
            } break;
        case LLM_ARCH_MPT:
            {
                result = llm.build_mpt();
            } break;
         case LLM_ARCH_STABLELM:
            {
                result = llm.build_stablelm();
            } break;
        case LLM_ARCH_QWEN:
            {
                result = llm.build_qwen();
            } break;
        case LLM_ARCH_QWEN2:
            {
                result = llm.build_qwen2();
            } break;
        case LLM_ARCH_QWEN2VL:
            {
                result = llm.build_qwen2vl();
            } break;
        case LLM_ARCH_QWEN2MOE:
            {
                result = llm.build_qwen2moe();
            } break;
        case LLM_ARCH_QWEN3:
            {
                result = llm.build_qwen3();
            } break;
        case LLM_ARCH_QWEN3MOE:
            {
                result = llm.build_qwen3moe();
            } break;
        case LLM_ARCH_MELLUM:
            {
                result = llm.build_mellum();
            } break;
        case LLM_ARCH_QWEN4EXP:
            {
                result = llm.build_qwen4exp();
            } break;
        case LLM_ARCH_QWEN3NEXT:
            {
                result = llm.build_qwen3next();
            } break;
        case LLM_ARCH_QWEN35MOE:
            {
                result = llm.build_qwen35moe();
            } break;
        case LLM_ARCH_QWEN35:
            {
                result = llm.build_qwen35();
            } break;
        case LLM_ARCH_QWEN3VL:
            {
                result = llm.build_qwen3vl();
            } break;
        case LLM_ARCH_QWEN3VLMOE:
            {
                result = llm.build_qwen3vlmoe();
            } break;
        case LLM_ARCH_PHI2:
            {
                result = llm.build_phi2();
            } break;
        case LLM_ARCH_PHI3:
            {
                result = llm.build_phi3();
            } break;
        case LLM_ARCH_PLAMO:
            {
                result = llm.build_plamo();
            } break;
        case LLM_ARCH_GPT2:
            {
                result = llm.build_gpt2();
            } break;
        case LLM_ARCH_CODESHELL:
            {
                result = llm.build_codeshell();
            } break;
        case LLM_ARCH_ORION:
            {
                result = llm.build_orion();
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                result = llm.build_internlm2();
            } break;
        case LLM_ARCH_MINICPM:
            {
                result = llm.build_minicpm();
            } break;
        case LLM_ARCH_GEMMA:
            {
                result = llm.build_gemma();
            } break;
        case LLM_ARCH_GEMMA2:
            {
                result = llm.build_gemma2();
            } break;
        case LLM_ARCH_GEMMA3:
            {
                result = llm.build_gemma3();
            } break;
        case LLM_ARCH_GEMMA_EMBEDDING:
            {
                result = llm.build_gemma_embedding();
            } break;
        case LLM_ARCH_GEMMA4:
            {
                result = llm.build_gemma4();
            } break;
        case LLM_ARCH_DIFFUSION_GEMMA:
            {
                result = llm.build_diffusion_gemma();
            } break;
        case LLM_ARCH_GEMMA4_MTP:
        case LLM_ARCH_GEMMA4_ASSISTANT:
            {
                result = llm.build_gemma4_mtp();
            } break;
        case LLM_ARCH_DFLASH:
        case LLM_ARCH_DFLASH2:
        case LLM_ARCH_DFLASH_DRAFT:
            {
                result = llm.build_dflash();
            } break;
        case LLM_ARCH_STARCODER2:
            {
                result = llm.build_starcoder2();
            } break;
        case LLM_ARCH_MAMBA:
            {
                result = llm.build_mamba();
            } break;
        case LLM_ARCH_XVERSE:
            {
                result = llm.build_xverse();
            } break;
        case LLM_ARCH_COMMAND_R:
            {
                result = llm.build_command_r();
            } break;
        case LLM_ARCH_DBRX:
            {
                result = llm.build_dbrx();
            } break;
        case LLM_ARCH_OLMO:
            {
                result = llm.build_olmo();
            } break;
        case LLM_ARCH_OPENELM:
            {
                result = llm.build_openelm();
            } break;
        case LLM_ARCH_GPTNEOX:
            {
                result = llm.build_gptneox();
            } break;
        case LLM_ARCH_ARCTIC:
            {
                result = llm.build_arctic();
            } break;
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_MISTRAL4:
            {
                result = llm.build_deepseek2();
            } break;
        case LLM_ARCH_DEEPSEEK4:
            {
                result = llm.build_deepseek4();
            } break;
        case LLM_ARCH_DEEPSEEK41:
            {
                static const bool v41_separate = llama_hparams::is_v41_separate();
                result = v41_separate ? llm.build_deepseek41() : llm.build_deepseek4();
            } break;
        case LLM_ARCH_OPENPANGU:
            {
                result = llm.build_openpangu();
            } break;
        case LLM_ARCH_CHATGLM:
            {
                result = llm.build_chatglm();
            } break;
        case LLM_ARCH_GLM4:
            {
                result = llm.build_glm4();
            } break;
        case LLM_ARCH_GLM4_MOE:
            {
                result = llm.build_glm4_moe();
            } break;
        case LLM_ARCH_BITNET:
            {
                result = llm.build_bitnet();
            } break;
        case LLM_ARCH_BITNET_B158:
        case LLM_ARCH_BITNET_25:
            {
                result = llm.build_bitnet_158();
            } break;
        case LLM_ARCH_COHERE2:
            {
                result = llm.build_cohere2();
            } break;
        case LLM_ARCH_COHERE2_MOE:
            {
                result = llm.build_cohere2_moe();
            } break;
        case LLM_ARCH_T5:
            {
                if (lctx.is_encoding) {
                    result = llm.build_t5_encoder();
                } else {
                    result = llm.build_t5_decoder();
                }
            } break;
        case LLM_ARCH_T5ENCODER:
            {
                result = llm.build_t5_encoder();
            } break;
        case LLM_ARCH_JAIS:
            {
                result = llm.build_jais();
            } break;
        case LLM_ARCH_DOTS1:
            {
                result = llm.build_dots1();
            } break;
        case LLM_ARCH_ERNIE4_5:
        {
            result = llm.build_ernie4_5();
        } break;
        case LLM_ARCH_ERNIE4_5_MOE:
        {
            result = llm.build_ernie4_5_moe();
        } break;
        case LLM_ARCH_HUNYUAN_MOE:
            {
                result = llm.build_hunyuan_moe();
            } break;
        case LLM_ARCH_OPENAI_MOE:
            {
                result = llm.build_openai_moe();
            } break;
        case LLM_ARCH_BAILINGMOE2:
            {
                result = llm.build_bailingmoe2();
            } break;
        case LLM_ARCH_BAILINGMOE3:
            {
                result = llm.build_bailingmoe3();
            } break;
        case LLM_ARCH_GLM5NEXT:
            {
                result = llm.build_glm5next();
            } break;
        case LLM_ARCH_MINIMAX_M2:
            {
                result = llm.build_minimaxm2();
            } break;
        case LLM_ARCH_MINIMAX_M3:
            {
                result = llm.build_minimaxm3();
            } break;
        case LLM_ARCH_SMOLLM3:
            {
                result = llm.build_smollm3();
            } break;
        case LLM_ARCH_HYPERCLOVAX:
            {
                result = llm.build_hyperclovax();
            } break;
        case LLM_ARCH_MISTRAL3:
            {
                result = llm.build_mistral3();
            } break;
        case LLM_ARCH_MIMO2:
            {
                result = llm.build_mimo2();
            } break;
        case LLM_ARCH_SEED_OSS:
            {
                result = llm.build_seedoss();
            } break;
        case LLM_ARCH_STEP35:
            {
                result = llm.build_step35();
            } break;
        case LLM_ARCH_LAGUNA:
            {
                result = llm.build_laguna();
            } break;
        case LLM_ARCH_MODERNBERT:   // S69 #MODERNBERT pre-norm GeGLU encoder embedder
            {
                result = llm.build_modernbert();
            } break;
        case LLM_ARCH_KIMI_K3:      // S69 hybrid KDA + MLA, latent MoE, situ, cross-layer residual
            {
                result = llm.build_kimi_k3();
            } break;
        case LLM_ARCH_LFM2:         // S69 short-convolution layers interleaved with attention
        case LLM_ARCH_LFM2MOE:      // ik_llama #2530: same graph, sparse FFN past n_layer_dense_lead
            {
                result = llm.build_lfm2();
            } break;
        case LLM_ARCH_GRANITE_HYBRID: // S69 mamba2 layers interleaved with attention
            {
                result = llm.build_granite_hybrid();
            } break;
        case LLM_ARCH_EAGLE3:       // S69 #5/#32d EAGLE3 draft-head (restaged onto upstream rebase 2026-06-22)
            {
                result = llm.build_eagle3();
            } break;
        case LLM_ARCH_K2_HORIZON:
            {
                result = llm.build_k2horizon();
            } break;
        case LLM_ARCH_INKLING:
            {
                result = llm.build_inkling();
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (result == nullptr) {
        llm.free();
        return nullptr;
    }

    result->n_batch = llm.n_tokens;

    // add on pooling layer
    if (lctx.cparams.mtp_op_type == MTP_OP_NONE && (lctx.cparams.embeddings ||
        (lctx.model.hparams.nextn_predict_layers > 0 || lctx.model.mtp))) {
        result = llm.append_pooling(result);
    }

    llm.free();

#if IK_PRINT_TIMING
    auto tim2 = ggml_time_us();
    printf("%s(...): %d us\n", __func__, int(tim2-tim1));
#endif

    return result;
}

ggml_tensor * llm_build_context::build_std_attention(ggml_cgraph * gf, ggml_tensor * the_attn_norm,
        ggml_tensor * input, ggml_tensor * inp_pos, ggml_tensor * inp_out_ids, ggml_tensor * rope_factors_in,
        ggml_tensor * KQ_mask, ggml_tensor * sinks, ggml_tensor * inp_attn_scale, float KQ_scale, float f_attn_scale,
        int n_swa, int il, bool do_rope, bool add_graph_split, bool add_input, bool is_norm, bool is_multi,
        ggml_tensor * post_norm, int kv_il, float post_norm_eps, post_norm_data * pnd,
        ggml_tensor ** k_view, ggml_tensor ** v_view) {

    float freq_base_l  = n_swa > 0 ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
    // FIXED 2026-09-09, slice 31. The non-SWA arm below read hparams.rope_freq_scale_train while
    // the line directly above it reads cparams.rope_freq_base. Two adjacent arms of one statement
    // disagreeing about their source is what marks this an oversight rather than a choice.
    //
    // cparams is correct, and three independent things say so:
    //   src/llama.cpp:8831  cparams.rope_freq_scale IS hparams.rope_freq_scale_train unless the
    //                       user passed --rope-freq-scale -- so reading hparams DISCARDED that
    //                       override on every architecture reaching this function;
    //   src/llama.cpp:8868  forces cparams.rope_freq_scale = 1.0f when the rope scaling type is
    //                       NONE ("never scale if scaling type is none"), a decision this path
    //                       could not see;
    //   src/graphs/build_gemma4.cpp:191 and upstream's llama_model::get_rope_freq_scale both read
    //                       cparams for exactly this quantity.
    //
    // In the DEFAULT case this is a no-op -- cparams.rope_freq_scale is initialised from that same
    // hparams field. It differs only when someone explicitly asked for something else, which is
    // both when it matters and why it went unnoticed: the failure is a slightly wrong RoPE scale,
    // fluent output, no error.
    //
    // The SWA arm is deliberately unchanged: cparams has no SWA variant, so
    // hparams.rope_freq_scale_train_swa is the only correct source there.
    //
    // NOT RUNTIME-VERIFIED. See S69-BUILD-STD-ATTENTION-DROPS-ROPE-FREQ-SCALE-OVERRIDE-2026-09-09-01
    // for the A/B that would settle it, including its own positive control.
    float freq_scale_l = n_swa > 0 ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
    if (hparams.has_rope_freq_base_per_layer) {
        freq_base_l = hparams.rope_freq_base_per_layer[il];
    }
    int n_rot_l         = lctx.model.hparams.rope_n_rot(il);
    float ext_factor_l  = ext_factor;
    float attn_factor_l = attn_factor;
    float beta_fast_l   = beta_fast;
    float beta_slow_l   = beta_slow;
    if (model.arch == LLM_ARCH_LAGUNA && n_swa > 0) {
        ext_factor_l  = 0.0f;
        attn_factor_l = 1.0f;
        beta_fast_l   = 32.0f;
        beta_slow_l   = 1.0f;
    }
#ifdef GGML_USE_VULKAN
    constexpr bool use_f32_precision = true;
#else
    constexpr bool use_f32_precision = false;
#endif

    bool should_use_f32_precision = use_f32_precision
                                  ||  model.arch == LLM_ARCH_PHI2
                                  || model.arch == LLM_ARCH_PHI3
                                  || model.arch == LLM_ARCH_GPTNEOX
                                  || model.arch == LLM_ARCH_QWEN2
                                  || model.arch == LLM_ARCH_COHERE2
                                  || model.arch == LLM_ARCH_COHERE2_MOE
                                  || model.arch == LLM_ARCH_COMMAND_R
                                  || model.arch == LLM_ARCH_GLM4
                               //   || model.arch == LLM_ARCH_GLM4_MOE
                                  || model.arch == LLM_ARCH_MIMO2
                                  || model.arch == LLM_ARCH_K2_HORIZON;
                               // || (model.arch == LLM_ARCH_DEEPSEEK2 && q->ne[1] <= 8);

    if (!model.layers[il].wqkv && !model.layers[il].wqk && cparams.flash_attn &&
         model.layers[il].wq->extra && model.layers[il].wk->extra && model.layers[il].wv->extra && model.layers[il].wo->extra) {
        if (kv_self.k_l[il]->extra && kv_self.v_l[il]->extra) {

            const bool compacted = kv_self.is_compacted(il);
            const bool use_swa_window = compacted && lctx.swa_window_view.active;
            const int32_t store_head = compacted ? swa_head : kv_head;
            const int32_t n_kv_view = use_swa_window ? (int32_t) lctx.swa_window_view.w_view : n_kv;
            const int32_t kv_view_offset = use_swa_window ? (int32_t) lctx.swa_window_view.win_off : 0;

            auto wq = (ggml_split_tensor_t *)model.layers[il].wq->extra;
            auto wk = (ggml_split_tensor_t *)model.layers[il].wk->extra;
            auto wv = (ggml_split_tensor_t *)model.layers[il].wv->extra;
            auto wo = (ggml_split_tensor_t *)model.layers[il].wo->extra;
            GGML_ASSERT(wq->n_device == wk->n_device && wq->n_device == wv->n_device && wq->n_device == wo->n_device);
            auto kl = (ggml_split_tensor_t *)kv_self.k_l[il]->extra;
            auto vl = (ggml_split_tensor_t *)kv_self.v_l[il]->extra;
            GGML_ASSERT(wq->n_device == kl->n_device && wq->n_device == vl->n_device);
            ggml_split_tensor_t *bq = nullptr, *bo = nullptr, *bk = nullptr, *bv = nullptr;
            if (model.layers[il].bq && model.layers[il].bq->extra) {
                bq = (ggml_split_tensor_t *)model.layers[il].bq->extra;
                GGML_ASSERT(bq->n_device == wq->n_device);
            }
            if (model.layers[il].bo && model.layers[il].bo->extra) {
                bo = (ggml_split_tensor_t *)model.layers[il].bo->extra;
                GGML_ASSERT(bo->n_device == wq->n_device);
            }
            if (model.layers[il].bk && model.layers[il].bk->extra) {
                bk = (ggml_split_tensor_t *)model.layers[il].bk->extra;
                GGML_ASSERT(bk->n_device == wq->n_device);
            }
            if (model.layers[il].bv && model.layers[il].bv->extra) {
                bv = (ggml_split_tensor_t *)model.layers[il].bv->extra;
                GGML_ASSERT(bv->n_device == wq->n_device);
            }
            std::vector<ggml_tensor*> attn(wq->n_device, nullptr);
            bool output_bias_added = false;
            int last_id = -1;
            for (int id = 0; id < wq->n_device; ++id) {
                int il_cb = 1000*(id+1) + il;
                auto split_wq = wq->splits[id];
                auto split_wk = wk->splits[id];
                auto split_wv = wv->splits[id];
                auto split_wo = wo->splits[id];
                auto split_kl = kl->splits[id];
                auto split_vl = vl->splits[id];
                GGML_ASSERT((!split_wq && !split_wk && !split_wv && !split_wo && !split_kl && !split_vl) ||
                        (split_wq && split_wk && split_wv && split_wo && split_kl && split_vl));
                if (!split_wq) continue;
                auto cur = get_input_tensor_sm_graph(ctx0, input, id);
                if (pnd) {
                    cur = llm_do_split_post_norm(ctx0, cur, pnd, id, wq->n_device, "ffn_post_norm", il_cb, cb);
                }
                cur = do_split_norm(ctx0, cur, the_attn_norm, lctx.model.hparams, cb, id, il_cb, is_norm);
                auto input_normed = cur;
                auto the_q_norm = model.layers[il].attn_q_norm ? model.layers[il].attn_q_norm->extra ?
                    ((ggml_split_tensor_t *)model.layers[il].attn_q_norm->extra)->splits[id] : model.layers[il].attn_q_norm : nullptr;
                auto the_k_norm = model.layers[il].attn_k_norm ? model.layers[il].attn_k_norm->extra ?
                    ((ggml_split_tensor_t *)model.layers[il].attn_k_norm->extra)->splits[id] : model.layers[il].attn_k_norm : nullptr;
                ggml_tensor *Qcur, *Kcur, *Vcur, *gate = nullptr;
                if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE || model.arch == LLM_ARCH_QWEN4EXP) {
                    auto [Q, K, V, G] = llm_build_mul_mat_qkv_gated(gf, cur, split_wq, split_wk, split_wv,
                            the_q_norm, the_k_norm, il);
                    Qcur = Q; Kcur = K; Vcur = V; gate = G;
                } else {
                    auto [Q, K, V] = llm_build_mul_mat_qkv(gf, cur, nullptr, nullptr, nullptr, nullptr,
                            split_wq, bq ? bq->splits[id] : nullptr,
                            split_wk, bk ? bk->splits[id] : nullptr,
                            split_wv, bv ? bv->splits[id] : nullptr,
                            the_q_norm, the_k_norm, f_attn_scale, il, add_graph_split);
                    Qcur = Q; Kcur = K; Vcur = V;
                    if (model.arch == LLM_ARCH_MIMO2 && std::abs(model.hparams.f_attn_v_scale - 1) > 1e-4f) {
                        Vcur = ggml_scale(ctx0, Vcur, model.hparams.f_attn_v_scale);
                        cb(Vcur, "Vcur_scales", il_cb);
                    }
                }
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                auto rope_factors = rope_factors_in;
                if (rope_factors) {
                    GGML_ASSERT(rope_factors->extra);
                    rope_factors = ((ggml_split_tensor_t *)rope_factors->extra)->splits[id];
                    GGML_ASSERT(rope_factors);
                }
                else if (model.layers[il].rope_freqs && model.layers[il].rope_freqs->extra) {
                    auto extra = (ggml_split_tensor_t *)model.layers[il].rope_freqs->extra;
                    rope_factors = extra->splits[id];
                }
                if (do_rope) {
                    if (is_multi) {
                        int sections[4];
                        std::copy(hparams.rope_sections.begin(), hparams.rope_sections.begin() + GGML_MROPE_SECTIONS, sections);
                        Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, rope_factors,
                                n_rot_l, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
                        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, rope_factors,
                                n_rot_l, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
                    } else {
                        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
                        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
                        ggml_build_forward_expand(gf, Qcur);
                        ggml_build_forward_expand(gf, Kcur);
                    }
                }
                cb(Qcur, "Qcur", il_cb);
                cb(Kcur, "Kcur", il_cb);
                if (inp_attn_scale) {
                    Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                    cb(Qcur, "Qcur_temp_scaled", il_cb);
                }
                if (cparams.k_cache_hadamard) {
                    if (int block_size = lctx.model.hadamard_size_k(il); block_size > 0) {
                        Qcur = ggml_hadamard(ctx0, Qcur, block_size);
                        Kcur = ggml_hadamard(ctx0, Kcur, block_size);
                        cb(Qcur, "Qcur_hadamard", il_cb);
                        cb(Kcur, "Kcur_hadamard", il_cb);
                    }
                }
                if (cparams.v_cache_hadamard) {
                    if (int block_size = lctx.model.hadamard_size_v(il); block_size > 0) {
                        Vcur = ggml_hadamard(ctx0, Vcur, block_size);
                        cb(Vcur, "Vcur_hadamard", il_cb);
                    }
                }
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                ggml_build_forward_expand(gf, Vcur);

                const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
                const int64_t n_head_kv     = split_wk->ne[1] / n_embd_head_k;

                GGML_ASSERT(kv_self.size == cparams.n_ctx);

                auto idx = 2*wq->n_device*il + 2*id;
                GGML_ASSERT(idx+1 < (int)lctx.cache_copies.size());
                auto k_row_size = ggml_row_size(split_kl->type, n_embd_head_k);
                ggml_tensor * k_cache_view = ggml_view_2d(ctx0, split_kl, n_embd_head_k, n_tokens*n_head_kv,
                        k_row_size, k_row_size*n_head_kv*store_head);

                lctx.cache_copies[idx+0].cpy  = ggml_cpy(ctx0, Kcur, k_cache_view);
                lctx.cache_copies[idx+0].step = k_row_size*n_head_kv;

                // note: storing RoPE-ed version of K in the KV cache
                ggml_build_forward_expand(gf, lctx.cache_copies[idx+0].cpy);

                struct ggml_tensor * v_cache_view = nullptr;

                if (cparams.flash_attn) {
                    v_cache_view = ggml_view_1d(ctx0, split_vl, n_tokens*split_wv->ne[1],
                            store_head*ggml_row_size(split_vl->type, split_wv->ne[1]));
                    lctx.cache_copies[idx+1].step = ggml_row_size(split_vl->type, split_wv->ne[1]);
                } else {
                    // note: the V cache is transposed when not using flash attention
                    v_cache_view = ggml_view_2d(ctx0, split_vl, n_tokens, split_wv->ne[1],
                            (  n_ctx)*ggml_element_size(split_vl),
                            store_head*ggml_element_size(split_vl));
                    lctx.cache_copies[idx+1].step = ggml_element_size(split_vl);

                    // flatten before transposing -- see llm_build_kv_store: a 3-D per-head Vcur transposed as
                    // is lands every element in the wrong cell of this [n_tokens, width] view
                    if (Vcur->ne[0] != split_wv->ne[1] || Vcur->ne[2] != 1 || Vcur->ne[3] != 1) {   // not ggml_n_dims(): that is 2 for a 1-token [d, h, 1]
                        if (!ggml_is_contiguous(Vcur)) {
                            Vcur = ggml_cont(ctx0, Vcur);
                        }
                        Vcur = ggml_reshape_2d(ctx0, Vcur, split_wv->ne[1], n_tokens);
                    }
                    Vcur = ggml_transpose(ctx0, Vcur);
                }
                cb(v_cache_view, "v_cache_view", il_cb);

                lctx.cache_copies[idx+1].cpy  = ggml_cpy(ctx0, Vcur, v_cache_view);
                ggml_build_forward_expand(gf, lctx.cache_copies[idx+1].cpy);

                auto q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                cb(q, "q", il_cb);

                auto knb1 = k_row_size*n_head_kv;
                auto k = ggml_view_3d(ctx0, split_kl, n_embd_head_k, n_kv_view, n_head_kv,
                             knb1, k_row_size, kv_view_offset*knb1);
                cb(k, "k", il_cb);

                auto v_row_size = ggml_row_size(split_vl->type, n_embd_head_v);
                auto vnb1 = ggml_row_size(split_vl->type, wv->splits[id]->ne[1]);
                auto v = ggml_view_3d(ctx0, split_vl, n_embd_head_v, n_kv_view, n_head_kv,
                             vnb1, v_row_size, kv_view_offset*vnb1);
                cb(v, "v", il_cb);

                cur = ggml_flash_attn_ext(ctx0, q, k, v, KQ_mask, KQ_scale, hparams.f_max_alibi_bias,
                        hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
                cb(cur, "flash_attn", il_cb);
                if (model.layers[il].attn_sinks && model.layers[il].attn_sinks->extra) {
                    auto split = (ggml_split_tensor_t *)model.layers[il].attn_sinks->extra;
                    GGML_ASSERT(split->n_device == wq->n_device);
                    GGML_ASSERT(split->splits[id]);
                    ggml_flash_attn_ext_add_sinks(cur, split->splits[id]);
                } else {
                    ggml_flash_attn_ext_add_sinks(cur, sinks);
                }
                if (n_swa > 0 && !lctx.kv_self.cells_disordered && cparams.n_seq_max == 1) {
                    ((int32_t *)cur->op_params)[4] = n_swa;
                }
                // Some models produced NaNs/gibberish when FA is computed with f16 precision on CUDA
                if (should_use_f32_precision) {
                    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
                }

                if (cparams.v_cache_hadamard) {
                    if (int block_size = lctx.model.hadamard_size_v(il); block_size > 0) {
                        cur = ggml_hadamard(ctx0, cur, block_size);
                        cb(cur, "flash_attn_h", il_cb);
                    }
                }

                if (model.layers[il].wqkv_gate) {
                    auto wqkv_gate = (ggml_split_tensor_t *)model.layers[il].wqkv_gate->extra;
                    GGML_ASSERT(wqkv_gate && wqkv_gate->splits[id]);
                    auto gate = llm_build_lora_mm(lctx, ctx0, wqkv_gate->splits[id], input_normed);
                    if (model.arch == LLM_ARCH_LAGUNA) {
                        gate = ggml_softplus(ctx0, gate);
                    }
                    cb(gate, "attn_gate", il_cb);
                    int nh = split_wo->ne[0]/n_embd_head_v;
                    auto attn_3d = ggml_reshape_3d(ctx0, cur, n_embd_head_v, nh, n_tokens);
                    if (model.arch == LLM_ARCH_LAGUNA) {
                        // Laguna uses a softplus gate. XS.2 stores one gate per head,
                        // while M.1 stores one gate per attention output element.
                        if (gate->ne[0] == nh) {
                            auto gate_3d = ggml_reshape_3d(ctx0, gate, 1, nh, n_tokens);
                            cur = ggml_mul(ctx0, attn_3d, gate_3d);
                        } else {
                            GGML_ASSERT(gate->ne[0] == split_wo->ne[0]);
                            cur = ggml_reshape_2d(ctx0, cur, split_wo->ne[0], n_tokens);
                            cur = ggml_mul(ctx0, cur, gate);
                        }
                    } else {
                        if (gate->ne[0] == n_embd_head_v * nh) {
                            gate = ggml_sigmoid(ctx0, gate);
                            cb(gate, "gate", il_cb);
                            cur = ggml_reshape_2d(ctx0, cur, gate->ne[0], gate->ne[1]);
                            //gate = ggml_reshape_3d(ctx0, gate, cur->ne[0], cur->ne[1], cur->ne[2]);
                            cur = ggml_mul(ctx0, cur, gate);
                        } else {
                            auto gate_3d = ggml_reshape_3d(ctx0, gate, 1, nh, n_tokens);
                            cur = ggml_fused_mul_unary(ctx0, gate_3d, attn_3d, GGML_UNARY_OP_SIGMOID);
                        }
                    }
                    cb(attn_3d, "attn_gated_3d", il_cb);
                }

                cur = ggml_reshape_2d(ctx0, cur, split_wo->ne[0], n_tokens);
                cb(cur, "flash_attn_reshaped", il_cb);
                if (gate) {
                    if (false && cur->ne[1] == 1) { // we need to add GGML_UNARY_OP_SIGMOID to the ops supported by ggml_fused_mul_unary
                        cur = ggml_fused_mul_unary(ctx0, cur, gate, GGML_UNARY_OP_SIGMOID);
                    } else {
                        gate = ggml_sigmoid(ctx0, gate);
                        cb(gate, "gate", il_cb);
                        cur = ggml_mul(ctx0, cur, gate);
                    }
                    cb(cur, "qkv_gated", il_cb);
                }

                if (inp_out_ids) {
                    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                    cb(cur, "fa_get_rows", il_cb);
                    if (pnd) {
                        pnd->next_input[id] = ggml_get_rows(ctx0, pnd->next_input[id], inp_out_ids);
                    }
                }

                cur = llm_build_lora_mm(lctx, ctx0, split_wo, cur);
                if (lctx.model.arch == LLM_ARCH_GLM4 || lctx.model.arch == LLM_ARCH_GLM4_MOE) {
                    // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
                    ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
                }
                cb(cur, "kqv_wo", il_cb);
                if (!output_bias_added && bo) {
                    cur = ggml_add(ctx0, cur, bo->splits[id]);
                    cb(cur, "kqv_wo_biased", il_cb);
                    output_bias_added = true;
                }

                if (cur->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
                    cur = ggml_cast(ctx0, cur, lctx.cparams.reduce_type);
                }
                ggml_build_forward_expand(gf, cur);
                attn[id] = cur;
                last_id = id;
            }
            GGML_ASSERT(last_id >= 0);
            if (add_input) {
                if (inp_out_ids) {
                    input = ggml_get_rows(ctx0, input, inp_out_ids);
                    cb(input, "sainp_get_rows", il);
                }
                attn[last_id] = ggml_add(ctx0, attn[last_id], input);
                cb(attn[last_id], "attn_out_with_input", il);
            }

            auto cur = ggml_reduce(ctx0, attn.data(), wq->n_device, GGML_OP_ADD);
            ggml_build_forward_expand(gf, cur);
            cb(cur, "attn_combined", il);
            return cur;
        }
    }

    auto cur = input;
    if (the_attn_norm) {
        cur = llm_build_norm(ctx0, cur, hparams, the_attn_norm, NULL, is_norm ? LLM_NORM : LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);
    }
    // TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28): mask the attn-norm output BEFORE
    // Wq/Wk/Wv (and, for gated-QKV architectures, Wqkv_gate) consume it -- this tensor is
    // unconditionally materialized here regardless of which fused/unfused Q/K/V path runs below
    // (unlike the FFN case), so no route-forcing flag is needed; calibration just reads the
    // existing "attn_norm-<il>" cb name unmasked. teal_sparsity==0 (default) leaves this a no-op.
    // NOT wired: this single-device path only -- the multi-device split-tensor attention branch
    // above (wq->extra != nullptr) returns early and never reaches here, so -sm graph/attn runs
    // are unmasked for attn_in even with --teal-sparsity > 0 (untested combination, not this
    // session's CPU-only single-device scope).
    if (cparams.teal_sparsity > 0.0f && !cparams.teal_calibrate &&
        il >= 0 && (size_t) il < cparams.teal_thresholds_attn_in.size() &&
        cparams.teal_thresholds_attn_in[il] > 0.0f) {
        cur = teal_apply_mask(ctx0, cur, cparams.teal_thresholds_attn_in[il]);
        cb(cur, "teal_attn_in_masked", il);
    }
    auto input_normed = cur;

    ggml_tensor *Qcur, *Kcur, *Vcur, *gate = nullptr;
    if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE || model.arch == LLM_ARCH_QWEN4EXP) {
        auto [Q, K, V, G] = llm_build_mul_mat_qkv_gated(gf, cur, model.layers[il].wq, model.layers[il].wk, model.layers[il].wv,
                model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, il);
        Qcur = Q; Kcur = K; Vcur = V; gate = G;
    } else {
        auto [Q, K, V] = llm_build_mul_mat_qkv(gf, cur,
                model.layers[il].wqkv, model.layers[il].bqkv,
                model.layers[il].wqk,  model.layers[il].bqk,
                model.layers[il].wq,   model.layers[il].bq, model.layers[il].wk, model.layers[il].bk, model.layers[il].wv, model.layers[il].bv,
                model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, f_attn_scale, il);
        Qcur = Q; Kcur = K; Vcur = V;
        if (model.arch == LLM_ARCH_MIMO2 && std::abs(model.hparams.f_attn_v_scale - 1) > 1e-4f) {
            Vcur = ggml_scale(ctx0, Vcur, model.hparams.f_attn_v_scale);
            cb(Vcur, "Vcur_scales", il);
        }
        if (model.arch == LLM_ARCH_GEMMA4) {
            Vcur = ggml_reshape_3d(ctx0, Vcur, model.hparams.n_embd_head_v(il), model.hparams.n_head_kv(il), n_tokens);
            Vcur = ggml_rms_norm(ctx0, Vcur, model.hparams.f_norm_rms_eps);
        }
    }

    // K smoothing: after q/k norm, before RoPE. The loader creates the scales only for NORM/NEOX rope and
    // single-device layers, so for every other case these are nullptr and this is a no-op.
    Qcur = llm_k_smooth_apply(ctx0, Qcur, model.layers[il].k_smooth_q);
    Kcur = llm_k_smooth_apply(ctx0, Kcur, model.layers[il].k_smooth_k);

    if (do_rope) {
        if (is_multi) {
            int sections[4];
            std::copy(hparams.rope_sections.begin(), hparams.rope_sections.begin() + GGML_MROPE_SECTIONS, sections);
            Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, rope_factors_in,
                    n_rot_l, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
            Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, rope_factors_in,
                    n_rot_l, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        } else {
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors_in, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
            Kcur = ggml_rope_ext( ctx0, Kcur, inp_pos, rope_factors_in, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        }
    }
    cb(Qcur, "Qcur_roped", il);
    cb(Kcur, "Kcur_roped", il);

    if (inp_attn_scale) {
        Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
        cb(Qcur, "Qcur_temp_scaled", il);
    }

    if (auto wqkv_gate = model.layers[il].wqkv_gate; wqkv_gate != nullptr) {
        cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                nullptr, nullptr,
                Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, KQ_scale, cb, il, sinks, n_swa, kv_il,
                k_view, v_view, swa_head);
        cb(cur, "wqkv", il);
        auto gate = llm_build_lora_mm(lctx, ctx0, wqkv_gate, input_normed);
        if (model.arch == LLM_ARCH_LAGUNA) {
            gate = ggml_softplus(ctx0, gate);
        }
        cb(gate, "attn_gate", il);
        int n_head_l = hparams.n_head(il);
        auto attn_3d = ggml_reshape_3d(ctx0, cur, n_embd_head_v, n_head_l, n_tokens);
        if (model.arch == LLM_ARCH_LAGUNA) {
            // Laguna uses a softplus gate. XS.2 stores one gate per head,
            // while M.1 stores one gate per attention output element.
            if (gate->ne[0] == n_head_l) {
                auto gate_3d = ggml_reshape_3d(ctx0, gate, 1, n_head_l, n_tokens);
                cur = ggml_mul(ctx0, attn_3d, gate_3d);
            } else {
                GGML_ASSERT(gate->ne[0] == n_embd_head_v * n_head_l);
                cur = ggml_mul(ctx0, cur, gate);
            }
        } else {
            if (gate->ne[0] == n_head_l) {
                auto gate_3d = ggml_reshape_3d(ctx0, gate, 1, n_head_l, n_tokens);
                cur = ggml_fused_mul_unary(ctx0, gate_3d, attn_3d, GGML_UNARY_OP_SIGMOID);
            } else {
                GGML_ASSERT(gate->ne[0] == n_embd_head_v * n_head_l);
                gate = ggml_sigmoid(ctx0, gate);
                cur  = ggml_mul(ctx0, cur, gate);
                //cur = ggml_fused_mul_unary(ctx0, gate, cur, GGML_UNARY_OP_SIGMOID);
            }
        }
        cb(cur, "attn_gated_3d", il);
        cur = ggml_reshape_2d(ctx0, cur, n_embd_head_v * n_head_l, n_tokens);
        cb(cur, "attn_gated", il);
        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
        if (model.layers[il].bo) {
            cur = ggml_add(ctx0, cur, model.layers[il].bo);
        }
        cb(cur, "attn_out", il);
    } else {
        if (gate) {
            cur = llm_build_kv(ctx0, lctx, kv_self, gf, nullptr, nullptr,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, KQ_scale, cb, il, sinks, n_swa, kv_il,
                    k_view, v_view, swa_head);
            if (false && cur->ne[1] == 1) { // we need to add GGML_UNARY_OP_SIGMOID to the ops supported by ggml_fused_mul_unary
                cur = ggml_fused_mul_unary(ctx0, cur, gate, GGML_UNARY_OP_SIGMOID);
            } else {
                gate = ggml_sigmoid(ctx0, gate);
                cb(gate, "gate", il);
                cur = ggml_mul(ctx0, cur, gate);
            }
            cb(cur, "qkv_gated", il);
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            if (model.layers[il].bo) {
                cur = ggml_add(ctx0, cur, model.layers[il].bo);
            }
            cb(cur, "attn_out", il);
        } else {
            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, KQ_scale, cb, il, sinks, n_swa, kv_il,
                    k_view, v_view, swa_head);
        }
    }

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        cb(cur, "sa_get_rows", il);
        if (add_input) {
            input = ggml_get_rows(ctx0, input, inp_out_ids);
            cb(input, "sainp_get_rows", il);
        }
    }

    if (post_norm) {
        if (post_norm_eps > 0) {
            cur = ggml_fused_rms_norm(ctx0, cur, post_norm, post_norm_eps);
        } else {
            cur = llm_build_norm(ctx0, cur, hparams, post_norm, NULL, LLM_NORM_RMS, cb, il);
        }
        cb(cur, "sa_normed", il);
    }

    if (add_input) {
        cb(cur, "attn_out", il);
        cur = ggml_add(ctx0, cur, input);
    }

    return cur;
}

int32_t llama_model_n_nextn_layer(const llama_model * model) {
    return model ? model->hparams.nextn_predict_layers : 0;
}

ggml_cgraph * llm_build_context::new_graph_custom() {
    int max_nodes = lctx.max_nodes(n_tokens, n_kv);
    return ggml_new_graph_custom(ctx0, max_nodes, false);
}
