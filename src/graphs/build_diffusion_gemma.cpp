// S69 unsloth port -- DiffusionGemma: block text-diffusion MoE on a Gemma-4 backbone.
//
// Ported from unsloth's llama.cpp `src/models/diffusion-gemma.cpp` (849 lines). That file is written
// against upstream's class-based graph layer (llm_graph_context / llm_graph_input_attn_no_cache /
// llama_model_base), NONE of which exists in this fork -- so this is a translation into
// llm_build_context, not a copy. See temp/research/findings/S69_LAND_DIFFUSION_GEMMA_2026-09-02.md.
//
// THE ARCHITECTURE, and the thing a careless port gets wrong: the encoder (causal prompt prefill)
// and the decoder (bidirectional canvas denoising) share EVERY weight except a per-layer
// layer_scalar. There is exactly one attention+FFN stack. A port that treats the two passes as
// separate weight sets still builds and still generates -- and is wrong.
//
// A single no-cache bidirectional forward over [prompt | canvas] reproduces the two-pass
// (causal encoder prefill + bidirectional decoder denoise, zero self-conditioning) result. Three
// things are region-aware, split at P = n_tokens - canvas_length (canvas = the LAST canvas_length
// positions):
//   1. input embeddings: prompt = embed*sqrt(n_embd); canvas = rms_norm_noscale(same) [+ SC]
//   2. per-layer scalar: prompt = enc_out_scale;      canvas = out_scale
//   3. attention mask  : prompt causal over prompt only; canvas bidirectional over prompt+canvas
// The Gemma-4 backbone itself is identical to build_gemma4() in graphs/build_gemma4.cpp.
//
// Three phases, selected by llama_diffusion_set_phase():
//   DG_PKV_UNIFIED  no-cache forward over [prompt | canvas]        (default AND safety fallback)
//   DG_PKV_PREFILL  forward a prompt chunk, writing its per-layer K,V into the prompt-KV store
//   DG_PKV_DECODE   forward the canvas only, reading the cached prompt K,V
// The region-aware masks themselves are filled host-side in llama_set_inputs() (src/llama.cpp),
// which is where this fork keeps every KQ_mask fill.

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------------------------
// lazily-allocated device-side stores (all hang off the const llama_model &, hence the mutable
// members declared in llama-model.h)
// ---------------------------------------------------------------------------------------------

static ggml_backend_buffer_type_t dg_layer0_buft(const llama_model & m) {
    if (!m.buft_layer.empty() && m.buft_layer[0].buft != nullptr) {
        return m.buft_layer[0].buft;
    }
    return ggml_backend_cpu_buffer_type();
}

// Build the SC soft-embedding weight once: tok_embd dequantized + transposed to [n_vocab, n_embd]
// F16 in a device weights buffer, so the per-step SC matmul runs on-device instead of on the CPU.
static void dg_ensure_sc_embT(const llama_model & m) {
    if (m.dg_sc_embT != nullptr) {
        return;
    }
    ggml_tensor * src = m.tok_embd;
    GGML_ASSERT(src != nullptr);
    const int64_t n_embd  = src->ne[0];
    const int64_t n_vocab = src->ne[1];

    ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, /*.no_alloc =*/ true };
    m.dg_sc_embT_ctx = ggml_init(ip);
    GGML_ASSERT(m.dg_sc_embT_ctx != nullptr);
    m.dg_sc_embT = ggml_new_tensor_2d(m.dg_sc_embT_ctx, GGML_TYPE_F16, n_vocab, n_embd);
    ggml_set_name(m.dg_sc_embT, "dg_sc_embT");

    m.dg_sc_embT_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.dg_sc_embT_ctx, dg_layer0_buft(m));
    GGML_ASSERT(m.dg_sc_embT_buf != nullptr);
    ggml_backend_buffer_set_usage(m.dg_sc_embT_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // dequantize each row on the host and scatter the transpose into F16
    const ggml_type st       = src->type;
    const size_t    row_size = ggml_row_size(st, n_embd);
    std::vector<char> host_src((size_t) row_size * n_vocab);
    ggml_backend_tensor_get(src, host_src.data(), 0, host_src.size());

    std::vector<ggml_fp16_t> dstT((size_t) n_vocab * n_embd);
    const ggml_type_traits_t tr = ggml_internal_get_type_traits(st);

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));
    auto worker = [&](int64_t v0, int64_t v1) {
        std::vector<float> tmp(n_embd);
        for (int64_t v = v0; v < v1; ++v) {
            const char * row = host_src.data() + (size_t) v * row_size;
            if (st == GGML_TYPE_F32) {
                std::memcpy(tmp.data(), row, (size_t) n_embd * sizeof(float));
            } else {
                tr.to_float(row, tmp.data(), n_embd);
            }
            for (int64_t e = 0; e < n_embd; ++e) {
                dstT[(size_t) e * n_vocab + v] = ggml_fp32_to_fp16(tmp[e]);
            }
        }
    };
    std::vector<std::thread> pool;
    const int64_t chunk = (n_vocab + nth - 1) / nth;
    for (unsigned t = 0; t < nth; ++t) {
        const int64_t v0 = (int64_t) t * chunk;
        const int64_t v1 = std::min(v0 + chunk, n_vocab);
        if (v0 < v1) {
            pool.emplace_back(worker, v0, v1);
        }
    }
    for (auto & th : pool) {
        th.join();
    }

    ggml_backend_tensor_set(m.dg_sc_embT, dstT.data(), 0, dstT.size() * sizeof(ggml_fp16_t));
}

// Lazily (re)allocate the device prev-step canvas-logits buffer [n_vocab, C] F32 (grow-only).
// Zero-init so step 0 (SC gated off) reads finite values: soft_max(0) = uniform, x 0 gate, no NaN.
static void dg_ensure_sc_dev(const llama_model & m, int64_t C) {
    const int64_t n_vocab = m.tok_embd->ne[1];
    if (m.dg_sc_dev != nullptr && m.dg_sc_dev_C >= C) {
        return;
    }
    if (m.dg_sc_dev_buf) { ggml_backend_buffer_free(m.dg_sc_dev_buf); m.dg_sc_dev_buf = nullptr; }
    if (m.dg_sc_dev_ctx) { ggml_free(m.dg_sc_dev_ctx); m.dg_sc_dev_ctx = nullptr; }

    ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, /*.no_alloc =*/ true };
    m.dg_sc_dev_ctx = ggml_init(ip);
    GGML_ASSERT(m.dg_sc_dev_ctx != nullptr);
    m.dg_sc_dev = ggml_new_tensor_2d(m.dg_sc_dev_ctx, GGML_TYPE_F32, n_vocab, C);
    ggml_set_name(m.dg_sc_dev, "dg_sc_dev");

    m.dg_sc_dev_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.dg_sc_dev_ctx, dg_layer0_buft(m));
    GGML_ASSERT(m.dg_sc_dev_buf != nullptr);
    ggml_backend_buffer_clear(m.dg_sc_dev_buf, 0);  // step-0 safety (see above)
    m.dg_sc_dev_C = C;
}

// Lazily (re)allocate the device-resident prompt-KV store (per-layer K,V, grow-only) for a prompt of
// length P at element type `type`, on layer 0's buffer type. Called from the PREFILL graph, where
// the type can follow cparams.flash_attn -- F16 halves the store and is precision-neutral under FA
// because flash attention casts K,V to F16 anyway.
static void dg_ensure_pkv_store(const llama_model & m, int64_t P, ggml_type type) {
    if (m.dg_pkv_buf != nullptr && m.dg_pkv_cap >= P && !m.dg_pkv_k.empty() && m.dg_pkv_k[0]->type == type) {
        return;
    }
    if (m.dg_pkv_buf) { ggml_backend_buffer_free(m.dg_pkv_buf); m.dg_pkv_buf = nullptr; }
    if (m.dg_pkv_ctx) { ggml_free(m.dg_pkv_ctx); m.dg_pkv_ctx = nullptr; }
    m.dg_pkv_k.clear();
    m.dg_pkv_v.clear();

    const int     n_layer = (int) m.hparams.n_layer;
    const int64_t cap     = P;

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (2 * n_layer + 4),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    m.dg_pkv_ctx = ggml_init(ip);
    GGML_ASSERT(m.dg_pkv_ctx != nullptr);
    m.dg_pkv_k.resize(n_layer);
    m.dg_pkv_v.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        const int64_t hd  = m.hparams.n_embd_head_k(il);
        const int64_t nkv = m.hparams.n_head_kv(il);
        m.dg_pkv_k[il] = ggml_new_tensor_3d(m.dg_pkv_ctx, type, hd, nkv, cap);
        m.dg_pkv_v[il] = ggml_new_tensor_3d(m.dg_pkv_ctx, type, hd, nkv, cap);
        ggml_format_name(m.dg_pkv_k[il], "dg_pkv_k_l%d", il);
        ggml_format_name(m.dg_pkv_v[il], "dg_pkv_v_l%d", il);
    }

    m.dg_pkv_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.dg_pkv_ctx, dg_layer0_buft(m));
    GGML_ASSERT(m.dg_pkv_buf != nullptr);
    m.dg_pkv_cap = cap;
}

void llama_model::dg_free_buffers() {
    if (dg_pkv_buf)     { ggml_backend_buffer_free(dg_pkv_buf);     dg_pkv_buf     = nullptr; }
    if (dg_pkv_ctx)     { ggml_free(dg_pkv_ctx);                    dg_pkv_ctx     = nullptr; }
    if (dg_sc_embT_buf) { ggml_backend_buffer_free(dg_sc_embT_buf); dg_sc_embT_buf = nullptr; }
    if (dg_sc_embT_ctx) { ggml_free(dg_sc_embT_ctx);                dg_sc_embT_ctx = nullptr; }
    if (dg_sc_dev_buf)  { ggml_backend_buffer_free(dg_sc_dev_buf);  dg_sc_dev_buf  = nullptr; }
    if (dg_sc_dev_ctx)  { ggml_free(dg_sc_dev_ctx);                 dg_sc_dev_ctx  = nullptr; }
    dg_pkv_k.clear();
    dg_pkv_v.clear();
    dg_pkv_cap  = 0;
    dg_sc_embT  = nullptr;
    dg_sc_dev   = nullptr;
    dg_sc_dev_C = 0;
}

// ---------------------------------------------------------------------------------------------
// the graph
// ---------------------------------------------------------------------------------------------

ggml_cgraph * llm_build_context::build_diffusion_gemma() {
    const llama_model & m = model;

    const int  phase      = m.dg_pkv_phase;
    const bool is_prefill = (phase == llama_model::DG_PKV_PREFILL);
    const bool is_decode  = (phase == llama_model::DG_PKV_DECODE);

    const int64_t canvas_length = (int64_t) hparams.diffusion_canvas_length;

    // Region split P|C. UNIFIED: prompt = first (n_tokens - canvas_length), canvas = last
    // canvas_length (P=0 for tiny warmup graphs). PREFILL: all prompt. DECODE: all canvas, and P is
    // the CACHED prompt length rather than anything in this batch.
    int64_t P, C;
    if (is_decode) {
        P = m.dg_pkv_P;
        C = n_tokens;
    } else if (is_prefill) {
        P = n_tokens;
        C = 0;
    } else {
        P = (canvas_length > 0 && n_tokens > canvas_length) ? (n_tokens - canvas_length) : 0;
        C = n_tokens - P;
    }

    // PREFILL processes one prompt chunk of n_tokens queries starting at global position
    // prefill_off; its keys span [0, prefill_off + n_tokens) -- prior chunks live in the store.
    const int64_t prefill_off = is_prefill ? m.dg_pkv_prefill_off : 0;

    // Allocate the store on the first PREFILL chunk, sized to the WHOLE prompt (dg_pkv_P).
    if (is_prefill && m.dg_pkv_P > 0) {
        dg_ensure_pkv_store(m, m.dg_pkv_P, flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32);
    }
    // A misuse must fail loudly rather than read out of bounds.
    if (is_prefill || is_decode) {
        const int64_t need = is_prefill ? (prefill_off + n_tokens) : P;
        GGML_ASSERT(!m.dg_pkv_k.empty() && !m.dg_pkv_v.empty() && m.dg_pkv_cap >= need &&
                    "DiffusionGemma prompt-KV store not allocated/sized for this phase");
    }

    ggml_cgraph * gf = new_graph_custom();

    ggml_tensor * cur;
    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, m.tok_embd, cb);
    cb(inpL, "tok_embd", -1);
    inpL = ggml_scale(ctx0, inpL, sqrtf((float) n_embd)); // ScaledWordEmbedding: embed_scale = sqrt(n_embd)
    cb(inpL, "inp_scaled", -1);

    // Canvas input embedding = rms_norm_noscale(embed*sqrt(n_embd) [+ self-conditioning]).
    // Zero SC -> the exactness forward. Shared by UNIFIED canvas rows and the DECODE batch.
    auto dg_canvas_embed = [&](ggml_tensor * canvas) -> ggml_tensor * {
        if (m.dg_sc_enabled && m.sc_pre_norm && m.sc_gate && m.sc_up && m.sc_down) {
            const int64_t Cc      = canvas->ne[1];
            const int64_t n_vocab = m.tok_embd->ne[1];
            canvas = ggml_cont(ctx0, canvas);

            ggml_tensor * sc_logits;
            if (m.dg_sc_device_resident) {
                // read the persistent device buffer the PRIOR step's lm_head wrote into (see the
                // ggml_cpy at the end of this function). Bit-identical to the host upload path.
                dg_ensure_sc_dev(m, Cc);
                sc_logits = m.dg_sc_dev;
            } else {
                sc_logits = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, Cc);
                ggml_set_input(sc_logits);
                cb(sc_logits, "dg_sc_logits", -1);
                lctx.inp_dg_sc_logits = sc_logits;
            }

            // raw/temperature, then softmax over the vocab (fp32)
            ggml_tensor * probs = ggml_soft_max(ctx0, ggml_scale(ctx0, sc_logits, m.dg_sc_temp_inv));
            dg_ensure_sc_embT(m);
            ggml_tensor * soft = ggml_mul_mat(ctx0, m.dg_sc_embT, probs);            // [n_embd, Cc]
            soft = ggml_scale(ctx0, soft, sqrtf((float) n_embd));
            // SC gated MLP: pre_norm -> down( gelu(gate) * up )
            ggml_tensor * normed = llm_build_norm(ctx0, soft, hparams, m.sc_pre_norm, nullptr, LLM_NORM_RMS, cb, -1);
            ggml_tensor * g  = ggml_gelu(ctx0, ggml_mul_mat(ctx0, m.sc_gate, normed));  // [n_ff, Cc]
            ggml_tensor * u  = ggml_mul_mat(ctx0, m.sc_up, normed);                     // [n_ff, Cc]
            ggml_tensor * sc = ggml_mul_mat(ctx0, m.sc_down, ggml_mul(ctx0, g, u));     // [n_embd, Cc]
            sc = ggml_scale(ctx0, sc, m.dg_sc_use);   // runtime {0,1} gate (0 == a block's first step)
            canvas = ggml_add(ctx0, canvas, sc);
            canvas = ggml_rms_norm(ctx0, canvas, hparams.f_norm_rms_eps); // post_norm, no scale
        } else {
            canvas = ggml_rms_norm(ctx0, canvas, hparams.f_norm_rms_eps); // no scale (zero-SC)
        }
        return canvas;
    };

    if (is_prefill) {
        // prompt-only (encoder): the scaled embedding feeds the layers directly, no canvas norm
    } else if (is_decode) {
        inpL = dg_canvas_embed(inpL);                       // canvas-only (decoder)
    } else if (P > 0 && P < n_tokens) {
        ggml_tensor * prompt = ggml_view_2d(ctx0, inpL, n_embd, P, inpL->nb[1], 0);
        ggml_tensor * canvas = ggml_view_2d(ctx0, inpL, n_embd, C, inpL->nb[1], P * inpL->nb[1]);
        canvas = dg_canvas_embed(ggml_cont(ctx0, canvas));
        inpL = ggml_concat(ctx0, ggml_cont(ctx0, prompt), canvas, 1);
    } else {
        inpL = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);   // pure-canvas (no prompt)
    }
    cb(inpL, "inp_region", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    // Region-aware no-cache masks. The FILL lives in llama_set_inputs(); here we only size them.
    //   UNIFIED : square [n_tokens, n_tokens]  (prompt causal, canvas bidirectional)
    //   DECODE  : rectangular [P+C, C]         (cached prompt K,V + fresh canvas K,V)
    //   PREFILL : rectangular [off+n_q, n_q]   (causal over this chunk + prior chunks)
    const int64_t n_kv_mask = is_decode  ? (P + C)
                            : is_prefill ? (prefill_off + n_tokens)
                                         : n_tokens;
    auto make_mask = [&](ggml_tensor ** slot, const char * name) -> ggml_tensor * {
        *slot = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv_mask, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        cb(*slot, name, -1);
        ggml_set_input(*slot);
        return flash_attn ? ggml_cast(ctx0, *slot, GGML_TYPE_F16) : *slot;
    };
    ggml_tensor * KQ_mask     = make_mask(&lctx.inp_KQ_mask, "KQ_mask");
    ggml_tensor * KQ_mask_swa = hparams.n_swa > 0 ? make_mask(&lctx.inp_KQ_mask_swa, "KQ_mask_swa") : KQ_mask;

    // Unlike autoregressive gemma4 there is no MID-STACK row selection: every canvas row's hidden
    // state feeds the next layer. The selection happens once, after the final norm (below).
    // It is REQUIRED: the PREFILL phase marks a single row for logits (the prompt's logits are
    // unused), so n_outputs < n_tokens and llama_set_inputs asserts on a missing inp_out_ids.
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = m.layers[il];

        const bool  is_sliding   = hparams.is_swa_impl[il] != 0;
        const float freq_base_l  = is_sliding ? hparams.rope_freq_base_train_swa  : cparams.rope_freq_base;
        const float freq_scale_l = is_sliding ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
        const int   n_rot_l      = is_sliding ? hparams.n_rot_swa : hparams.n_rot;
        const int64_t n_embd_head_l = hparams.n_embd_head_k(il);
        const int64_t n_head_l      = hparams.n_head(il);
        const int64_t n_head_kv_l   = hparams.n_head_kv(il);
        const int64_t n_embd_gqa_l  = n_embd_head_l * n_head_kv_l;

        GGML_ASSERT(n_embd_head_l == hparams.n_embd_head_v(il));

        // full_attention (non-SWA) layers use rope_freqs for proportional rope
        ggml_tensor * freq_factors = is_sliding ? nullptr : layer.rope_freqs;
        ggml_tensor * KQ_mask_l    = is_sliding ? KQ_mask_swa : KQ_mask;

        cur = llm_build_norm(ctx0, inpL, hparams, layer.attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // ---- Q/K/V + q/k-norm + partial rope: the shared Gemma-4 backbone -----------------
        ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, layer.wq, cur);
        cb(Qcur, "Qcur", il);
        ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, layer.wk, cur);
        cb(Kcur, "Kcur", il);
        // global layers have no v_proj -> value reuses the raw k_proj output
        ggml_tensor * Vcur = layer.wv ? llm_build_lora_mm(lctx, ctx0, layer.wv, cur) : Kcur;
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_l, n_head_l,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_l, n_head_kv_l, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_l, n_head_kv_l, n_tokens);

        Qcur = llm_build_norm(ctx0, Qcur, hparams, layer.attn_q_norm, nullptr, LLM_NORM_RMS, cb, il);
        Kcur = llm_build_norm(ctx0, Kcur, hparams, layer.attn_k_norm, nullptr, LLM_NORM_RMS, cb, il);
        Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);   // v-norm: no scale
        cb(Qcur, "Qcur_normed", il);
        cb(Kcur, "Kcur_normed", il);
        cb(Vcur, "Vcur_normed", il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);
        cb(Kcur, "Kcur_pos", il);

        // ---- prompt-KV store: write on PREFILL, read on PREFILL(prefix)/DECODE ------------
        // The store is F16 under FA (halves it; FA casts K,V to F16 anyway, so it is
        // precision-neutral) and F32 otherwise. The fresh K,V stay F32: the store WRITE is an
        // F32->store-type cpy, but the prefix READ is cast back to F32 because the CUDA concat op
        // is F32-only. cast_kv() is a no-op when the store is already F32.
        const auto cast_kv = [&](ggml_tensor * t) {
            return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx0, t, GGML_TYPE_F32);
        };

        ggml_tensor * Kfull = Kcur;
        ggml_tensor * Vfull = Vcur;
        int64_t       n_kv_l = n_tokens;

        if (is_prefill) {
            ggml_tensor * sk_all = m.dg_pkv_k[il];
            ggml_tensor * sv_all = m.dg_pkv_v[il];
            const size_t koff = (size_t) prefill_off * sk_all->nb[2];
            const size_t voff = (size_t) prefill_off * sv_all->nb[2];
            ggml_tensor * sk = ggml_view_3d(ctx0, sk_all, n_embd_head_l, n_head_kv_l, n_tokens,
                                            sk_all->nb[1], sk_all->nb[2], koff);
            ggml_tensor * sv = ggml_view_3d(ctx0, sv_all, n_embd_head_l, n_head_kv_l, n_tokens,
                                            sv_all->nb[1], sv_all->nb[2], voff);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, sk));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, sv));
            if (prefill_off > 0) {
                ggml_tensor * pk = ggml_view_3d(ctx0, sk_all, n_embd_head_l, n_head_kv_l, prefill_off,
                                                sk_all->nb[1], sk_all->nb[2], 0);
                ggml_tensor * pv = ggml_view_3d(ctx0, sv_all, n_embd_head_l, n_head_kv_l, prefill_off,
                                                sv_all->nb[1], sv_all->nb[2], 0);
                Kfull = ggml_concat(ctx0, cast_kv(pk), Kcur, 2);
                Vfull = ggml_concat(ctx0, cast_kv(pv), Vcur, 2);
            }
            n_kv_l = prefill_off + n_tokens;
        } else if (is_decode) {
            ggml_tensor * sk_all = m.dg_pkv_k[il];
            ggml_tensor * sv_all = m.dg_pkv_v[il];
            ggml_tensor * pk = ggml_view_3d(ctx0, sk_all, n_embd_head_l, n_head_kv_l, P,
                                            sk_all->nb[1], sk_all->nb[2], 0);
            ggml_tensor * pv = ggml_view_3d(ctx0, sv_all, n_embd_head_l, n_head_kv_l, P,
                                            sv_all->nb[1], sv_all->nb[2], 0);
            Kfull = ggml_concat(ctx0, cast_kv(pk), Kcur, 2);
            Vfull = ggml_concat(ctx0, cast_kv(pv), Vcur, 2);
            n_kv_l = P + C;
        }
        GGML_ASSERT(n_kv_l == n_kv_mask && "DiffusionGemma mask width must match the K,V length");

        // ---- attention (no KV cache; explicit K,V + region-aware additive mask) -----------
        ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);                        // [hd, n_tokens, n_head]
        ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kfull, 0, 2, 1, 3));      // [hd, n_kv, n_head_kv]

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);                                   // [n_kv, n_tokens, n_head]
        cb(kq, "kq", il);
        kq = ggml_soft_max_ext(ctx0, kq, KQ_mask_l, hparams.f_attention_scale, hparams.f_max_alibi_bias);
        cb(kq, "kq_soft_max_ext", il);

        ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0,
                              ggml_reshape_2d(ctx0, Vfull, n_embd_gqa_l, n_kv_l)));    // [n_kv, n_embd_gqa]
        v = ggml_reshape_3d(ctx0, v, n_kv_l, n_embd_head_l, n_head_kv_l);

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);                                 // [hd, n_tokens, n_head]
        ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_head_l * n_head_l, n_tokens);
        cb(cur, "kqv_merged_cont", il);
        ggml_build_forward_expand(gf, cur);

        cur = llm_build_lora_mm(lctx, ctx0, layer.wo, cur);
        cb(cur, "kqv_out", il);

        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // ---- dense MLP (shared expert) + 128-expert MoE: the shared Gemma-4 backbone ------
        if (layer.ffn_gate_inp) {
            ggml_tensor * cur_mlp = llm_build_norm(ctx0, attn_out, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(cur_mlp, "ffn_norm_1", il);
            cur_mlp = llm_build_ffn(ctx0, lctx, nullptr, cur_mlp,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_GELU, LLM_FFN_PAR, cb, il, gf);

            // MoE. The router operates on the UNNORMED post-attention residual attn_out:
            // rms_norm(no scale) -> *1/sqrt(n_embd) -> *ffn_gate_inp_s -> gate proj -> softmax top-k.
            // The 1/sqrt(n_embd) factor is NOT in this graph on purpose: llm_scale_gate_inp_s()
            // (llama.cpp) folds it into ffn_gate_inp_s at load time for GEMMA4 and DIFFUSION_GEMMA,
            // exactly as build_gemma4.cpp does. Do not add a ggml_scale here - it would double-apply.
            ggml_tensor * cur_moe = llm_build_norm(ctx0, attn_out, hparams, layer.ffn_pre_norm_2, nullptr, LLM_NORM_RMS, cb, il);
            cb(cur_moe, "ffn_norm_2", il);
            ggml_tensor * tmp = ggml_fused_rms_norm(ctx0, attn_out, layer.ffn_gate_inp_s, hparams.f_norm_rms_eps);
            cb(tmp, "ffn_moe_router_in", il);
            ggml_tensor * logits = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_inp, tmp);
            cb(logits, "ffn_moe_logits", il);

            cur_moe = llm_build_moe_ffn(ctx0, lctx, cur_moe,
                    nullptr,                                  // gate_inp (logits passed in below)
                    layer.ffn_up_gate_exps ? nullptr : layer.ffn_up_exps,
                    layer.ffn_up_gate_exps ? nullptr : layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    nullptr,                                  // exp_probs_b
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true, false, 0.0f,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    cb, il, gf, false,
                    layer.ffn_up_gate_exps, nullptr, logits, layer.ffn_down_exps_s);

            cur = ggml_fused_rms_rms_add(ctx0, cur_mlp, layer.ffn_post_norm_1,
                                               cur_moe, layer.ffn_post_norm_2, hparams.f_norm_rms_eps);
            cb(cur, "ffn_moe_combined", il);
            ggml_build_forward_expand(gf, cur);

            cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_post_norm, NULL, LLM_NORM_RMS, cb, -1);
            cb(cur, "ffn_post_norm", il);
            cur = ggml_add(ctx0, cur, attn_out);
        } else {
            cur = llm_build_ffn(ctx0, lctx, layer.ffn_norm, attn_out,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_GELU, LLM_FFN_PAR, cb, il, gf,
                    true, false, nullptr, layer.ffn_post_norm);
            cb(cur, "ffn_out", il);
        }

        // ---- REGION-AWARE per-layer scalar -----------------------------------------------
        // This is the ONLY weight that differs between the encoder and decoder passes. Applying
        // the decoder scalar to the prompt rows compiles, runs, and produces wrong logits.
        if (is_prefill) {
            cur = ggml_mul(ctx0, cur, layer.enc_out_scale);
        } else if (is_decode) {
            cur = ggml_mul(ctx0, cur, layer.out_scale);
        } else if (P > 0 && P < n_tokens) {
            ggml_tensor * prompt = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, P, cur->nb[1], 0));
            ggml_tensor * canvas = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, C, cur->nb[1], P * cur->nb[1]));
            prompt = ggml_mul(ctx0, prompt, layer.enc_out_scale);
            canvas = ggml_mul(ctx0, canvas, layer.out_scale);
            cur = ggml_concat(ctx0, prompt, canvas, 1);
        } else {
            cur = ggml_mul(ctx0, cur, layer.out_scale);
        }
        cb(cur, "out_scaled", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, m.output_norm, NULL, LLM_NORM_RMS, cb, -1);

    // keep only the requested output rows. When every row is requested this is the identity
    // permutation; in PREFILL it drops the whole prompt's lm_head work.
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, m.output, cur);

    if (hparams.f_final_logit_softcapping > 0) {
        cb(cur, "result_pre_softcap", -1);
        cur = ggml_softcap(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping, hparams.f_final_logit_softcapping);
    }
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    // Device-resident SC: persist THIS step's canvas logits (the last C rows) into dg_sc_dev so the
    // NEXT step reads them on-device with no host round-trip. The cpy is downstream of the SC read
    // in the DAG, so topological order runs read-before-write on the single buffer (no hazard), and
    // F32->F32 makes it the exact bytes the host path would have uploaded.
    if (m.dg_sc_enabled && m.dg_sc_device_resident && C > 0) {
        dg_ensure_sc_dev(m, C);
        const int64_t n_out   = cur->ne[1];
        const int64_t n_vocab = cur->ne[0];
        GGML_ASSERT(n_out >= C && "device SC expects the canvas rows to be present in the output logits");
        ggml_tensor * canvas_logits = ggml_view_2d(ctx0, cur, n_vocab, C, cur->nb[1], (n_out - C) * cur->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, canvas_logits, m.dg_sc_dev));
    }

    return gf;
}
