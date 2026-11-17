#pragma once

#include "llama-impl.h"

#include <cstdint>
#include <array>
#include <vector>
#include <bitset>
#include <cassert>
#include <cmath>

#define LLAMA_MAX_LAYERS  512
#define LLAMA_MAX_PLE_NGRAM 8
#define LLAMA_MAX_PLE_HEADS 64

// ADDED 2026-09-08, adopted from 279b4bfaf^2:src/llama-hparams.h:11 with upstream's own value and
// its own comment. Read by the merged src/llama-graph.cpp:2274; this header declared no such
// bound, so that file would not compile:
//     error: 'LLAMA_MAX_EXPERTS' was not declared in this scope; did you mean 'LLAMA_MAX_LAYERS'?
// The value is not arbitrary -- 1024 is what Kimi K3 needs, and a smaller bound would silently
// truncate its expert set rather than fail.
#define LLAMA_MAX_EXPERTS 1024 // Kimi K3
#define LLAMA_MAX_PLE_NGRAM 8  // qwen4exp, deepseek41 (all hash modules)
#define LLAMA_MAX_PLE_HEADS 64 // qwen4exp, deepseek41 (all hash modules)

// ADDED 2026-09-08, adopted verbatim from 279b4bfaf^2:src/llama-hparams.h:21-26.
//
// The merge dropped this enum entirely while keeping every one of its users, so the type was
// referenced 16 times and declared nowhere. It is the single most-repeated compile error left in
// src/ -- 9 translation units, llama-graph.cpp and every llama-kv-cache*.cpp among them, all
// stopping on the same line:
//     src/llama-kv-cache.h:111:16: error: 'llama_swa_type' has not been declared
// All four members were already in use in this tree (LLAMA_SWA_TYPE_NONE / _STANDARD / _CHUNKED /
// _SYMMETRIC), which is how the drop is known to be a drop rather than a feature the fork never
// had: you cannot reference the members of an enum you never adopted.
//
// Values are explicit and match upstream's, because hparams.swa_type is written by GGUF-driven
// loaders in src/models/ and read by mask construction in llama-graph.cpp -- a renumbering here
// would change which tokens get masked, silently, with no build error anywhere.
// ADDED 2026-09-08, copied from 279b4bfaf^2:src/llama-hparams.h:29.
// The enum itself lives in src/llama-graph.h:53 on BOTH sides -- this fork already has it,
// identical. Upstream forward-declares it here with its fixed underlying type so the hparams field
// below can be declared without pulling in llama-graph.h, and that is exactly what is wanted here:
// llama-hparams.h is included BY llama-graph.h, so the reverse include would be circular.
// Upstream enums the merged kv-cache and graph code read (merged 2026-11-04). llama_swa_type and
// llama_expert_gating_func_type are defined further down in this file by the fork, so they are not repeated.
// how the non-causal mask should be constructed with llama_set_causal_attn(ctx, false)
// (e.g. mtmd decoding image tokens)
enum llama_non_causal_type {
    LLAMA_NON_CAUSAL_TYPE_ALL      = 0, // all layers non-causal, SWA still applied (gemma 3, qwen-vl, ...)
    LLAMA_NON_CAUSAL_TYPE_SWA_ONLY = 1, // SWA layers non-causal, dense layers stay causal (gemma 4)
    LLAMA_NON_CAUSAL_TYPE_SWA_FULL = 2, // all layers non-causal, SWA not applied between tokens of the current ubatch (deepseek 4)
};

// M-RoPE: which input position slot feeds each RoPE section
enum llama_rope_section_order {
    LLAMA_ROPE_SECTION_ORDER_UNSPECIFIED = -1,
    LLAMA_ROPE_SECTION_ORDER_TYXZ        = 0, // default, slot i feeds section i
    LLAMA_ROPE_SECTION_ORDER_ZYXT        = 1, // MiniCPM-V 4.7: time last
};


// forward declaration; full definition in llama-graph.h
enum llm_ffn_op_type : int;

// ADDED 2026-09-08, adopted verbatim from 279b4bfaf^2:src/llama-hparams.h.
// Two tiny carrier structs for the WavTokenizer decoder's PosNet and ConvNeXt stacks. They are
// separate types rather than flat fields because each stack has its own depth and width, and
// src/models/wavtokenizer-dec.cpp reads them as hparams.posnet.n_embd / .n_layer at 9 sites and
// hparams.convnext.* at 5 -- so flattening them would mean editing those call sites for no gain.
struct llama_hparams_posnet {
    uint32_t n_embd;
    uint32_t n_layer;
};

struct llama_hparams_convnext {
    uint32_t n_embd;
    uint32_t n_layer;
};

enum llama_swa_type {
    LLAMA_SWA_TYPE_NONE      = 0,
    LLAMA_SWA_TYPE_STANDARD  = 1,
    LLAMA_SWA_TYPE_CHUNKED   = 2,
    LLAMA_SWA_TYPE_SYMMETRIC = 3,
};

// ADDED 2026-09-08: upstream's spelling of this enum, which the merge left undefined.
//
// The merge took upstream's src/llama-graph.h and .cpp, which name the type
// `llama_expert_gating_func_type` and the constants `LLAMA_EXPERT_GATING_FUNC_TYPE_*` -- and left
// this header providing only the fork's older `llm_expert_gating_func_type` / `LLM_*` spelling.
// So the BUILT graph code referenced a type that existed nowhere:
//     src/llama-graph.h:1111 and :1136   error: 'llama_expert_gating_func_type' has not been declared
//     src/llama-graph.cpp:1909, :1957    same parameter
//     src/llama-graph.cpp:1989, :1993    case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX / _SIGMOID
// Same family as the CC_* -> GGML_CUDA_CC_* rename fixed elsewhere in this merge: the header came
// from one side and its users from the other. That one failed silently in the preprocessor; this
// one is a hard compile error, which is the better of the two failure modes.
//
// The two enums are VALUE-IDENTICAL, 0..4 with the same meanings, verified against
// 279b4bfaf^2:src/llama-hparams.h:13-19 line by line. So upstream's names become canonical and the
// fork's older spellings are kept as ALIASES rather than replaced -- src/llama-hparams.h:79's
// `expert_gating_func = LLM_EXPERT_GATING_FUNC_SOFTMAX` and every other existing use keeps
// compiling unchanged. Note the fork's set is internally inconsistent (two members omit `_TYPE`),
// which is a good reason to prefer upstream's as the one to write new code against.
enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_NONE           = 0,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX        = 1,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID        = 2,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = 3, // applied to the router weights, not the logits
    LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS  = 4,
};

// The fork's historical spelling, preserved so no existing caller changes. Each member is defined
// as the upstream constant it equals, so the two can never drift apart silently.
enum llm_expert_gating_func_type {
    LLM_EXPERT_GATING_FUNC_TYPE_NONE           = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE,
    LLM_EXPERT_GATING_FUNC_SOFTMAX             = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
    LLM_EXPERT_GATING_FUNC_SIGMOID             = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
    LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
    LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS  = LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS,
};

struct llama_hparams {
    bool vocab_only;

    // ADDED 2026-09-08, adopted from 279b4bfaf^2:src/llama-hparams.h:46, where upstream declares it
    // on the very next line after vocab_only. Read by src/llama-kv-cache.cpp:323 and :734, which
    // could not compile without it.
    //
    // It is WIRED, not just declared: src/llama.cpp sets it from llama_model_params.no_alloc right
    // beside the existing vocab_only assignment. Before that wiring this fork had a DEAD SWITCH --
    // include/llama.h:545 declares the parameter, common/fit.cpp:56 sets it to true to get a
    // metadata-only load for its memory breakdown, and nothing in src/ read it (measured: 0
    // word-boundary hits in src/, against 30 for use_mmap as a positive control). The caller asked
    // for a dry run and got a real allocation.
    //
    // STILL NOT HONOURED after this change, named so nobody reads it as complete: the model WEIGHT
    // load path (upstream llama-model.cpp:1701,1723,1781, via llama_model_loader::no_alloc at
    // upstream llama-model-loader.h:82 -- this fork's loader has no such member) and the
    // context/graph-reserve path (upstream llama-context.cpp:485,634,653,668,677,3271). Only the
    // KV-cache half is reachable here.
    bool no_alloc = false;   // only load metadata and simulate memory allocations
    bool rope_finetuned;
    bool use_par_res;

    uint32_t n_vocab;
    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_embd_out = 0;
    uint32_t n_layer;

    // ADOPTED FROM UPSTREAM 2026-09-08 (279b4bfaf^2:src/llama-hparams.h:56 and :214). Neither is
    // in the fork parent, so nothing of ours is displaced -- these are two upstream hparams that
    // arrived with upstream's accessors (llama_model_n_layer_nextn, llama_model_n_cls_out) while
    // the fields themselves did not. Upstream's DEFAULTS are kept deliberately: n_cls_out is 1,
    // not 0, because it counts classifier outputs and a zero default would silently produce a
    // model that reports no outputs rather than the single default one.
    uint32_t n_layer_nextn = 0;
    uint32_t n_cls_out     = 1;
    int32_t n_layer_kv_from_start = -1; // if non-negative, the first n_layer_kv_from_start layers have KV cache
    uint32_t n_rot;
    uint32_t n_rot_swa;
    uint32_t n_swa = 0; // sliding window attention (SWA)
    uint32_t n_swa_pattern = 1; // by default, all layers use non-sliding-window attention
    // ADDED 2026-09-08 (279b4bfaf^2:src/llama-hparams.h:157). WHICH KIND of sliding window, as
    // opposed to n_swa's HOW WIDE. Written by the GGUF-driven loaders in src/models/ and read by
    // mask construction in llama-graph.cpp (13 sites) and the kv-cache classes (4). Every one of
    // those readers survived the merge; only the field and its enum did not, which is why nine
    // translation units stopped at "'llama_swa_type' has not been declared".
    llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;
    bool     swa_full_non_causal = false; // DSV4 vision: in a non-causal (image) span, window only clips older tokens
    uint32_t n_layer_decision = 0; // trailing blocks that form the decision head (lfm2, modern-bert)
    uint32_t n_embd_head_k_full; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v_full; // dimension of values (d_v) aka n_embd_head
    uint32_t n_embd_head_k_swa;
    uint32_t n_embd_head_v_swa;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_vocab_type = 0; // for BERT-style token types
    uint32_t n_rel_attn_bkts = 0;

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;
    // hy_v4 (slice 47c). Upstream's declaration verbatim, identical in
    // type to n_ff_arr above.
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_exp_arr; // per-layer expert feed-forward size (upstream's second copy removed)
    // per-layer top-k expert routing count
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_expert_used_arr;

    uint32_t n_layer_dense_lead = 0;
    uint32_t n_lora_q           = 0;
    uint32_t n_lora_kv          = 0;
    uint32_t n_ff_exp           = 0;
    // DiffusionGemma: canvas occupies the LAST diffusion_canvas_length positions of the batch;
    // everything before it is the prompt. 0 on every other arch.
    uint32_t diffusion_canvas_length = 0;
    uint32_t n_ff_shexp         = 0;
    uint32_t n_expert_shared    = 0;
    // K2 Horizon MoVA
    uint32_t n_value_expert      = 0;
    uint32_t n_value_expert_used = 0;

    uint32_t n_norm_groups      = 0;
    uint32_t n_expert_groups    = 0;
    uint32_t n_group_used       = 0;
    uint32_t n_group_experts    = 0;

    float    expert_group_scale   = 0.05f;
    float    expert_weights_scale = 0.0f;
    // S69 kimi-k3 (field names match mainline llama.cpp)
    uint32_t n_expert_latent      = 0;      // latent MoE bottleneck width
    uint32_t attn_res_block_size  = 0;      // cross-layer residual block size
    float    situ_beta            = 0.0f;   // situ activation, attention path
    float    situ_linear_beta     = 0.0f;   // situ activation, linear (KDA) path
    bool     expert_weights_norm  = false;
    uint32_t expert_gating_func   = LLM_EXPERT_GATING_FUNC_SOFTMAX;
    uint32_t moe_every_n_layers   = 0;
    uint32_t nextn_predict_layers = 0;

    // ADDED 2026-09-09. THE TWO FIELDS ABOVE HOLD ONE QUANTITY AND ARE LOADED FROM ONE GGUF KEY.
    // Measured over all 334 .h/.hpp/.cpp under src/: LLM_KV_NEXTN_PREDICT_LAYERS is read at 29
    // sites -- 20 write n_layer_nextn (every one in src/models/) and 9 write nextn_predict_layers
    // (every one inside llm_load_hparams()) -- and ZERO code lines mention both, so nothing
    // reconciled them. Whichever loader ran decided which field held the value; the other kept its
    // initialiser, 0, for the whole run. n_layer_eff() reads only nextn_predict_layers, so a model
    // whose count landed in the other field would report the TOTAL layer count where the EFFECTIVE
    // one was meant -- the same defect class fork commit 4477ee8b fixed across 138 files.
    //
    // It has been harmless so far ONLY because load_arch_hparams has zero call sites, so just the
    // 9 in-tree loaders ever run. An architecture dispatcher -- the next step of this merge -- is
    // exactly what would make the divergence live.
    //
    // These make the two agree by construction. Two directional helpers rather than one setter,
    // because ml.get_key() writes THROUGH A REFERENCE to whichever field the call site named, so
    // the value is not available to pass in and the sync must read back the field just written.
    //
    // NOT a rename and NOT a deletion, deliberately: n_layer_nextn has 110 reads across 26 files
    // and nextn_predict_layers 109 across 17, and the two names do not always mean the same thing
    // to their callers -- src/models/gemma4-assistant.cpp:15 asserts n_layer_nextn == n_layer and
    // iterates it as the FULL layer range (an MTP-only assistant head), while
    // src/models/deepseek4.cpp:28 asserts it is < n_layer and subtracts it to find the main trunk.
    // Making the STORAGE agree is separable from deciding what the NAME means, and only the first
    // is a precondition for the dispatcher.
    // Rows: S69-NEXTN-TWO-FIELDS-ONE-GGUF-KEY-2026-09-09-01,
    //       S69-NEXTN-LAYER-COUNT-TWO-DEFECTS-THAT-CANCEL-2026-09-09-01.
    void sync_nextn_from_n_layer_nextn()        { nextn_predict_layers = n_layer_nextn; }
    void sync_nextn_from_nextn_predict_layers() { n_layer_nextn = nextn_predict_layers; }
    bool nextn_fields_agree() const { return n_layer_nextn == nextn_predict_layers; }

    float f_norm_eps;
    float f_norm_rms_eps;
    float f_norm_group_eps;

    // post norms use their own epsilon
    float f_post_norm_eps = 0.0f;

    float f_attn_logit_softcapping   = 50.0f;
    float f_router_logit_softcapping = 30.0f;
    float f_final_logit_softcapping  = 30.0f;

    float    rope_attn_factor = 1.0f;
    float    rope_freq_base_train;
    float    rope_freq_base_train_swa;
    float    rope_freq_scale_train;
    float    rope_freq_scale_train_swa;
    uint32_t rope_scaling_apply_mask = 0x1;
    bool     has_rope_freq_base_per_layer = false;
    uint32_t n_ctx_orig_yarn;
    float    rope_yarn_log_mul = 0.0f;
    // HF rope_parameters["mscale"] (numerator of the YaRN attention-factor ratio; 0 = key absent, HF treats 0 as unset too)
    // and rope_parameters["truncate"] (HF default true: floor/ceil the YaRN correction range). Keys
    // <arch>.rope.scaling.yarn_mscale / yarn_truncate (row S69-EXO-YARN-TRUNCATE-FALSE-AND-MSCALE-RATIO-2026-11-06-01).
    float    rope_yarn_mscale  = 0.0f;
    bool     rope_yarn_truncate = true;

    float    yarn_ext_factor  = -1.0f;
    float    yarn_attn_factor =  1.0f;
    float    yarn_beta_fast   = 32.0f;
    float    yarn_beta_slow   =  1.0f;
    std::array<int, 4> rope_sections;
    std::array<float,    LLAMA_MAX_LAYERS> rope_freq_base_per_layer;
    std::array<uint32_t, LLAMA_MAX_LAYERS> rope_dim_per_layer;

    // M-RoPE: which input position slot feeds each RoPE section (llm_graph_input_pos)
    enum llama_rope_section_order rope_section_order = LLAMA_ROPE_SECTION_ORDER_TYXZ;

    // see llama_non_causal_type
    // note: for SWA_FULL, older tokens (outside the current ubatch) are still window-clipped
    llama_non_causal_type non_causal_type = LLAMA_NON_CAUSAL_TYPE_ALL;

    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    uint32_t ssm_n_group = 0;
    // Set for mamba2-style SSM layers (granitehybrid). MUST be tested before ssm_n_group in
    // n_embd_k_s()/n_embd_v_s(): mamba2 also sets ssm_n_group (=1), and those functions treat
    // a non-zero group count as "this is qwen3next", which would size a delta-net state.
    bool     ssm_mamba2  = false;
    // Set for the src/models classes with grouped mamba2 SSM layers (mamba2, falcon-h1, nemotron_h*): they are written against upstream's
    // TWO-tensor recurrent cache, r = conv state (d_conv-1)*(d_inner + 2*n_group*d_state) and s = ssm state d_state*d_inner. Without it a non-zero
    // ssm_n_group routes to the qwen3next sizing below, r gets width 0 and build_mamba2_layer's reshape of the conv state aborts.
    bool     ssm_split_state = false;
    // S69 LFM2: short-convolution cache length. d_conv = n_shortconv_l_cache - 1.
    uint32_t n_shortconv_l_cache = 0;
    // ADDED 2026-09-08 with upstream's defaults (279b4bfaf^2:src/llama-hparams.h:129, :181, :229,
    // :231). These four gate the architecture branches in n_embd_r()/n_embd_s() and
    // build_inp_attn_scale(); the merged src/llama-graph.cpp reads all of them and this struct
    // declared none.
    //
    // Every one defaults to the value that makes its branch INERT -- 0 for the three head-size /
    // scale fields -- except token_shift_count, which is 2 upstream because that is RWKV's actual
    // shift depth and it is only ever consulted when wkv_head_size != 0. So on every model this
    // tree loads today these change nothing, and on the day the src/models/ loaders that populate
    // them are wired, the branches are already right.
    //
    // FOLLOW-UP OWED: nothing in this tree WRITES wkv_head_size, n_embd_head_kda or n_embd_head_la
    // yet, because their loaders are in the unbuilt src/models/. A field nobody writes is a field
    // whose branch never runs -- said plainly here rather than left to look finished.
    uint32_t wkv_head_size          = 0;    // RWKV
    uint32_t n_embd_head_kda        = 0;    // Kimi KDA linear-attention head dim
    uint32_t n_embd_head_la         = 0;    // MiniMax-Text-01 linear-attention head dim
    uint32_t token_shift_count      = 2;    // RWKV token-shift depth
    bool     kda_safe_gate = false;
    float    kda_gate_lower_bound = 0.0f;

    // In-Place TTT (arXiv 2604.06169): on these layers ffn_down is a per-sequence fast weight updated chunk by
    // chunk, apply-then-update. Official code (Apache-2.0) ByteDance-Seed/In-Place-TTT be2324829b0e,
    // inference_model/hf_qwen3/modeling_qwen3.py:117-152 (MLP) and :345-377 (chunk carry).
    // ttt_chunk == 0 means the model has no TTT. Row S69-FORK-INPLACE-TTT-2026-10-26-01.
    std::array<bool, LLAMA_MAX_LAYERS> ttt_layer_impl = {};
    uint32_t ttt_chunk        = 0;
    float    ttt_lr           = 0.0f;
    uint32_t ttt_target       = 0;    // 0 = hidden_states (the FFN input), 1 = input_embed (token embeddings)
    uint32_t ttt_conv_padding = 0;    // 0 = symmetric k=5 pad 2 (official code), 1 = causal left pad 4 (paper text)
    float    ttt_clip_tau     = 0.0f; // 0 = no clip (official code); > 0 = Frobenius-norm clip of each dW (paper)

    bool is_ttt(uint32_t il) const { return ttt_chunk > 0 && il < n_layer && ttt_layer_impl[il]; }
    bool has_ttt() const {
        if (ttt_chunk == 0) return false;
        for (uint32_t il = 0; il < n_layer; ++il) if (ttt_layer_impl[il]) return true;
        return false;
    }

    // for hybrid state-space models (e.g. qwen3next)
    std::array<bool, LLAMA_MAX_LAYERS> is_recr_impl;

    // hrm-text (looped H/L stacks)
    uint32_t n_hrm_layers_per_stack = 0;
    uint32_t n_hrm_h_cycles = 0;
    uint32_t n_hrm_l_cycles = 0;
    bool     hrm_prefix_lm = false;

    float f_clamp_kqv      = 0.0f;
    float f_max_alibi_bias = 0.0f;
    float f_logit_scale    = 0.0f;

    // Additional scale factors (Granite/Granite MoE)
    float f_residual_scale  = 0.0f;
    float f_embedding_scale = 0.0f;
    float f_attention_scale = 0.0f;
    float f_attn_v_scale    = 1.0f;

    // grok-2
    float    f_attn_out_scale = 0.0f;
    uint32_t attn_temp_length = 0;

    bool causal_attn   = true;
    bool use_alibi     = false;
    bool attn_soft_cap = false;

    // LiLiCorr draft head (dflash.cpp)
    uint32_t lilicorr_hidden_size    = 0;
    uint32_t lilicorr_candidate_topk = 0;
    uint32_t lilicorr_num_heads      = 0;
    uint32_t lilicorr_layers         = 0;
    float    lilicorr_vector_eps     = 0.0f;
    float    lilicorr_logit_scale    = 0.0f;

    uint32_t n_moe_layer_step        = 0;
    bool     use_kq_norm             = true;
    uint32_t n_attn_chunk            = 0;
    // values below seems to be fixed on llama4
    uint32_t n_no_rope_layer_step    = 4;
    uint32_t n_attn_temp_floor_scale = 8192;
    float    f_attn_temp_scale       = 0.1;
    // ADDED 2026-09-08 (279b4bfaf^2:src/llama-hparams.h:231). The third argument of upstream's
    // attention-temperature input, read at src/llama-graph.cpp:2411 -- an OFFSET applied to the
    // position index before the floor/scale, where the two fields above are the floor and the
    // scale. This struct carried the other two and not this one, so build_inp_attn_scale() would
    // not compile. Upstream's default of 0.0f is the identity, so nothing this tree loads today
    // behaves differently.
    float    f_attn_temp_offset      = 0.0f;

    // DSA (deepseek sparse attention)
    uint32_t indexer_n_head     = 0;
    uint32_t indexer_head_size  = 0;
    uint32_t indexer_top_k      = 0;
    // k-pool indexer: tokens per compressed key cell (GGUF attention.indexer.kpool); 0 = plain per-token DSA (as in GLM-5.2)
    // (indexer_block_size is declared further down with the minimax-m3 indexer fields; GLM5NEXT reads it from the same key family)
    // GLM-5.2 IndexShare: per-layer full/shared indexer map. "full" layers compute their own lightning-
    // indexer top-k; "shared" layers reuse the previous full layer's top-k. Populated from GGUF
    // indexer_types metadata if present, else derived from the GLM-5.2 config rule at load time.
    // WIDENED 2026-09-09 from std::array<bool, ...>. This is a FORK-INTRODUCED NARROWING being
    // undone, not an upstream feature being adopted: upstream declares it
    //     279b4bfaf^2:src/llama-hparams.h:261  std::array<uint32_t, LLAMA_MAX_LAYERS> is_indexer_full_impl;
    // and src/models/glm-dsa.cpp:6 defines its default table as
    //     const std::array<uint32_t, LLAMA_MAX_LAYERS> GLM_5_2_DEFAULT_INDEXER_TYPES
    // so glm-dsa.cpp:68 could not assign one to the other. The note at :845 records that these three
    // fields were given upstream's SPELLINGS on purpose (is_swa_impl / is_recr_impl /
    // is_indexer_full_impl, 86 sites); the spelling was adopted and the TYPE was not, which also left
    // this field inconsistent with its own neighbour is_swa_impl, already uint32_t here.
    // is_indexer_full() below returns bool and indexes this array, and uint32_t converts to bool
    // there unchanged, so no call site moves.
    // The `= {}` is KEPT although upstream has no initialiser: an uninitialised per-layer array is
    // the same defect class as the indeterminate `mtp` flag fixed in f5a354bb, where tensor loading
    // was decided by reading uninitialised memory. Costs 1,536 bytes; still trivially copyable, so
    // the static_assert at the end of this struct still holds.
    std::array<uint32_t, LLAMA_MAX_LAYERS> is_indexer_full_impl = {};
    // use (1 + (pos + offset) / floor_scale) ^ scale instead of the llama4 log form
    bool     attn_temp_pow           = false;

    // realtime models: number of tokens the output lags behind the audio input
    uint32_t n_tokens_delay          = 0;

    // k-pool indexer (glm5-next, qwen4exp)
    uint32_t indexer_kpool     = 0; // k-pool size
    bool     indexer_kpool_select_tail = true;
    // head-size slots per cached indexer row, the last one holds the pooled key
    uint32_t indexer_kpool_row = 3;
    // pools are consecutive cells in sequence order, not runs of consecutive positions
    bool     indexer_kpool_by_order = false;

    // openPangu-2.0 (mHC / Hyper-Connections + learned param sink)
    uint32_t mhc_num_stream    = 1;
    uint32_t mhc_recur_norm    = 0;
    uint32_t param_sink_number = 0;
    // window used by the NextN/MTP layers in place of n_swa
    uint32_t n_swa_mtp = 0;

    // DeepSeek-V4
    uint32_t dsv4_o_group_count     = 0;
    uint32_t dsv4_o_lora_rank       = 0;
    uint32_t dsv4_hc_mult           = 0;
    uint32_t dsv4_hc_sinkhorn_iters = 0;
    uint32_t dsv4_hash_layer_count  = 0;
    float    dsv4_compress_rope_base = 0.0f;
    float    dsv4_hc_eps             = 0.0f;
    std::array<uint32_t, LLAMA_MAX_LAYERS> dsv4_compress_ratios = {};

    // The two compressed-stream plan slots. V4 hardcodes 4 (overlapping groups) and 128;
    // V4.1 takes both ratios from the file and pools disjoint groups in both.
    uint32_t dsv4_csa_ratio   = 4;
    uint32_t dsv4_hca_ratio   = 128;
    bool     dsv4_csa_overlap = true;
    bool     dsv4_hca_overlap = false;

    // DeepSeek-V4.1: only a few source layers compress, and the layers after each one read the
    // same rows. For every layer these hold the layer that published what it reads, or -1. A
    // layer whose entry is itself is a source. Index keys and the indexer top-k are shared the
    // same way on their own sets of layers.
    bool dsv4_shared_streams = false;
    // V4.1: the hyper-connection mix a sublayer computes is applied by the next one, and the
    // last FFN's mix collapses the output (no learned output head). V4 uses the same mix
    // twice and has a head.
    bool dsv4_hc_lag = false;
    // V4 re-normalizes every attention head after the up projection; V4.1 normalizes only the
    // low-rank query and the latent kv.
    bool dsv4_q_head_norm = true;
    std::array<int32_t, LLAMA_MAX_LAYERS> dsv41_kv_source        = {};
    std::array<int32_t, LLAMA_MAX_LAYERS> dsv41_index_key_source = {};
    std::array<int32_t, LLAMA_MAX_LAYERS> dsv41_topk_source      = {};

    int32_t  dsv4_candidate_source_layer = -1;
    uint32_t dsv4_candidate_block_size   = 0;
    uint32_t dsv4_candidate_topk_blocks  = 0;
    bool dsv41_is_kv_source   (uint32_t il) const { return dsv4_shared_streams && dsv41_kv_source[il]        == (int32_t) il; }
    bool dsv41_owns_index_k   (uint32_t il) const { return dsv4_shared_streams && dsv41_index_key_source[il] == (int32_t) il; }
    bool dsv41_is_index_source(uint32_t il) const { return dsv4_shared_streams && dsv41_topk_source[il]      == (int32_t) il; }

    // DeepSeek-V4.1 engram: n-gram keyed tables added into the stream at a few layers
    uint32_t engram_n_head         = 0;
    uint32_t engram_key_length     = 0;
    uint32_t engram_max_ngram_size = 0;
    uint32_t engram_n_layer        = 0;
    std::array<uint32_t, LLAMA_MAX_LAYERS> engram_layer_ids = {};

    // which engram table layer il carries, or -1
    int engram_index(uint32_t il) const {
        for (uint32_t e = 0; e < engram_n_layer; ++e) {
            if (engram_layer_ids[e] == il) return (int) e;
        }
        return -1;
    }

    // qwen4exp. hc_low_rank 0 means the full-rank hyper-connection form; the
    // ple_* group is inert unless the model carries an n-gram embedding layer.
    uint32_t hc_low_rank         = 0;
    // hy_v4 (slice 47c). Upstream's declaration verbatim, beside its
    // sibling above.
    float hc_magnitude = 0.0f;
    uint32_t ple_ngram_size      = 0;
    uint32_t ple_heads_per_ngram = 0;
    uint32_t ple_conv_kernel     = 0;
    uint32_t ple_n_heads         = 0;   // (ngram_size - 1) * heads_per_ngram
    uint32_t ple_head_dim        = 0;
    uint32_t ple_eos_token_id    = 0;
    uint32_t ple_image_token_id  = 0;
    std::array<bool,     LLAMA_MAX_LAYERS> ple_layer_arr = {};
    std::array<uint64_t, LLAMA_MAX_PLE_NGRAM> ple_layer_multipliers = {};
    std::array<uint64_t, LLAMA_MAX_PLE_HEADS> ple_head_offsets      = {};
    std::array<uint64_t, LLAMA_MAX_PLE_HEADS> ple_head_vocab_sizes  = {};
    bool     dsv4_hc_single_pass = false; // DeepSeek-V4.1 DSpark draft: lagged pre-mix, no hc_head

    // deepseek41
    std::array<int32_t, LLAMA_MAX_LAYERS> dsv41_kv_src;   // layer whose compressed KV this layer reads, -1 = none
    std::array<int32_t, LLAMA_MAX_LAYERS> dsv41_topk_src; // layer whose top-k this layer uses, -1 = none
    int32_t  dsv41_cand_src         = -1;    // -1 = no candidate block selection
    uint32_t dsv41_cand_topk_blocks = 0;
    uint32_t dsv41_cand_block_size  = 0;

    // (merge 2026-11-04: upstream's copies of hc_low_rank, hc_magnitude, ple_ngram_size .. ple_image_token_id and the
    //  three ple_* arrays were removed here -- the fork declares them above with value-initialisers and 64-bit
    //  ple_head_offsets / ple_head_vocab_sizes; the members below are upstream additions the fork did not have)
    // the id the KV cells record at embedding (image) positions for the n-gram history; LLAMA_TOKEN_NULL = none
    llama_token ple_embd_token   = LLAMA_TOKEN_NULL;
    // hash constant sets, stored back to back: 1 shared by all PLE layers (qwen4exp), or 1 per PLE layer in layer order (deepseek41)
    uint32_t ple_n_modules       = 1;
    // the file lists PLE layer indices, so this is never a per-layer gguf array and can hold one bit per layer
    std::bitset<LLAMA_MAX_LAYERS> is_ple_impl;
    // is_ple() / ple_conv_state(): the fork defines both further down (fork semantics: ple_conv_state() is
    // (kernel - 1) * ngram_size, n_embd_ple_conv() multiplies by dsv4_hc_mult * n_embd). Upstream's out-of-line
    // versions fold that product into ple_conv_state(); see the caller check recorded in the merge tracker.

    // qwen3vl deepstack
    uint32_t n_deepstack_layers = 0;

    // S69 #5/#30 — EAGLE3 draft-head hparams (PR #18039 port; n_embd_tgt==0 = not an EAGLE3 model).
    // Fixed array (not std::vector) so llama_hparams stays trivially-copyable (static_assert).
    std::array<int32_t, 4> target_layer_ids = { -1, -1, -1, -1 }; // EAGLE3 low/mid/high (use [0..2]); size 4 reuses the get_key_or_arr<array<int,4>> instantiation
    uint32_t n_embd_tgt           = 0;               // target model n_embd; n_embd_inp = 3 * n_embd_tgt
    uint32_t n_draft_vocab        = 0;               // EAGLE3 draft vocab size (output head); 0 = n/a
    bool     norm_before_residual = false;           // EAGLE3 residual-norm ordering

    // gemma4 per-layer embedding
    uint32_t n_embd_per_layer = 0;

    // gemma4 separate assistant MTP
    uint32_t mtp_backbone_n_embd = 0;
    bool     mtp_use_ordered_embeddings = false;
    uint32_t mtp_num_centroids = 0;
    uint32_t mtp_centroid_top_k = 0;

    // DFlash draft model metadata
    uint32_t dflash_block_size = 16;
    uint32_t dflash_mask_token_id = 0;
    uint32_t dflash_n_target_features = 0;
    uint32_t dflash_n_target_layers = 0;
    uint32_t dflash_target_layer_ids[8] = {};
    uint32_t dflash_conv_kernel_size = 0;
    uint32_t dflash_conv_group_size = 0;
    uint32_t dflash_selector_rank = 0;
    uint32_t dflash_selector_top_k = 0;
    float    dflash_backbone_rotary_base = 0.0f;
    bool     dflash_laguna = false;
    bool     dflash_dsv4 = false;
    // NOTE (ik-main slice S02): dflash_selector_top_k, the DFlash2 candidate-selector width read by
    // llama_model_dflash_selector_top_k (src/llama-ext.h; body from commit b10f9ca58, "spec : add DFlash2 support",
    // #27342), is declared ONCE, next to dflash_conv_* / dflash_selector_rank above (ik_llama #2345 declares the same
    // field; the two ports collided). 0 = model is not DFlash2; every GGUF that lacks the key keeps that.
    bool     dflash_dsv41 = false;  // DSV4 draft with V4.1 rules: lagged hyper-connections, no output head
    // DSpark proposal blocks attend bidirectionally within the block and to one fixed window
    // of the last n_swa committed positions (reference get_dspark_topk_idxs); other drafts
    // keep the causal, per-row sliding mask
    bool     dflash_block_bidir = false;

    // needed by encoder-decoder models (e.g. T5, FLAN-T5)
    // ref: https://github.com/ggerganov/llama.cpp/pull/8141
    llama_token dec_start_token_id = -1;

    enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;
    enum llama_pooling_type      pooling_type_cls        = LLAMA_POOLING_TYPE_UNSPECIFIED; // pooling before the classifier head (RANK)
    enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;

    std::array<uint32_t, LLAMA_MAX_LAYERS> is_swa_impl;

    std::array<float, LLAMA_MAX_LAYERS> swiglu_clamp_exp;
    std::array<float, LLAMA_MAX_LAYERS> swiglu_clamp_shexp;

    // ================================================================================
    // ADOPTED 2026-09-08 from 279b4bfaf^2:src/llama-hparams.h -- 41 fields, verbatim
    // (types, defaults and upstream's own comments), one documented exception noted below.
    //
    // WHY: src/models/*.cpp -- 150 files, this fork's whole model-architecture layer -- reference
    // these members at 266 sites. They were merged in against upstream's struct and this struct
    // never gained the members, so every one of those files stops at a missing declaration. That
    // is the real reason src/models/ is out of the build: a missing declaration, not a policy.
    //
    // NOT ADDED, because they already exist here under this fork's own spelling and the GGUF key
    // each loader reads proves they are the same fact (adding them would put two fields in this
    // struct holding one value, which is how three "losses" earlier today turned out to be
    // renames):
    //     upstream n_rot_full         == this fork's n_rot           (LLM_KV_ROPE_DIMENSION_COUNT)
    //     upstream n_embd_out_impl    == this fork's n_embd_out      (LLM_KV_EMBEDDING_LENGTH_OUT)
    //     upstream f_attn_value_scale == this fork's f_attn_v_scale  (LLM_KV_ATTENTION_VALUE_SCALE)
    //     upstream n_layer_all        == this fork's n_layer         (both the TOTAL layer count)
    // n_shortconv_l_cache is also skipped: this fork already declares it.
    //
    // DO NOT "align" the n_layer one by renaming this fork's n_layer to n_layer_all and adding
    // upstream's n_layer() method. Upstream's n_layer() returns the EFFECTIVE count
    // (n_layer_all - n_layer_nextn) while every existing use of the bare name in this tree means
    // the TOTAL. That rename changes the meaning of all of them, on MTP models only, with no
    // compiler error anywhere. This fork spells the effective count n_layer_eff().
    // ================================================================================

    // MLA (deepseek2 and friends): larger heads in MQA form, decompressed to MHA.
    uint32_t n_embd_head_k_mla_impl = 0;
    uint32_t n_embd_head_v_mla_impl = 0;
    uint32_t n_embd_head_k_mla_swa  = 0;
    uint32_t n_embd_head_v_mla_swa  = 0;
    uint32_t n_lora_kv_swa          = 0;

    // per-layer RoPE on/off pattern. NOTE: this is NOT this fork's rope_dim_per_layer -- that one
    // reads LLM_KV_ROPE_DIMENSION_COUNT_PER_LAYER and holds a DIMENSION per layer, while this
    // reads LLM_KV_ATTENTION_ROPE_PATTERN ("%s.attention.rope_pattern") and holds a flag. Two
    // different keys, two different facts; they were candidates for the same member and are not.
    std::array<uint32_t, LLAMA_MAX_LAYERS> rope_pattern;

    // qwen3-vl deepstack: explicit layer mapping when the implicit one is not enough
    std::array<int32_t, LLAMA_MAX_LAYERS> deepstack_mapping_arr;

    // chameleon.
    // Upstream declares this uninitialised; adopted as `= false` instead, deliberately. Every
    // comparable bool in this struct carries an explicit default, and an indeterminate bool read
    // by a graph builder is a real hazard. This is the only intentional divergence in this block.
    bool swin_norm = false;

    // eagle3
    bool norm_before_fc = false;

    // WavTokenizer
    struct llama_hparams_posnet   posnet;
    struct llama_hparams_convnext convnext;

    // RWKV / ARWKV
    uint32_t n_lora_decay           = 0;
    uint32_t n_lora_iclr            = 0;
    uint32_t n_lora_value_res_mix   = 0;
    uint32_t n_lora_gate            = 0;
    uint32_t time_mix_extra_dim     = 0;
    uint32_t time_decay_extra_dim   = 0;
    uint32_t rescale_every_n_layers = 0;

    // T5 and other encoder-decoder models: the decoder's own layer count
    uint32_t dec_n_layer = 0;

    // gemma-embedding dense projection stacks
    uint32_t dense_2_feat_in  = 0;  // in_features of the 2_Dense
    uint32_t dense_2_feat_out = 0;  // out_features of the 2_Dense
    uint32_t dense_3_feat_in  = 0;  // in_features of the 3_Dense
    uint32_t dense_3_feat_out = 0;  // out_features of the 3_Dense

    // grovemoe
    uint32_t n_ff_chexp = 0;

    // minimax-m3 DSA indexer
    uint32_t indexer_block_size   = 0;
    uint32_t indexer_local_blocks = 0;

    // modern-bert: which gated activation this arch's FFN uses. The enum is forward-declared at
    // the top of this header and defined in src/llama-graph.h:53. Left uninitialised exactly as
    // upstream has it -- an enumerator cannot be named here while the enum is incomplete.
    enum llm_ffn_op_type llm_ffn_op;
    enum ggml_unary_op   act_cls = GGML_UNARY_OP_TANH; // activation of the classifier head (RANK)

    // nemotron-h
    uint32_t moe_latent_size = 0;

    // apertus xIELU activation, per layer
    std::array<float, LLAMA_MAX_LAYERS> xielu_alpha_n;
    std::array<float, LLAMA_MAX_LAYERS> xielu_alpha_p;
    std::array<float, LLAMA_MAX_LAYERS> xielu_beta;
    std::array<float, LLAMA_MAX_LAYERS> xielu_eps;

    // eagle3 encoder input width (0 = derive it); read through n_embd_inp_enc()
    uint32_t n_embd_inp_enc_impl = 0;

    // explicit input embedding width (0 = derive it); read through n_embd_inp(), which this
    // adoption also patches -- see the note on that method.
    uint32_t n_embd_inp_impl = 0;

    // gemma3n altup / laurel
    uint32_t n_altup      = 4; // altup_num_inputs
    uint32_t n_embd_altup = 256;
    uint32_t laurel_rank  = 64;
    uint32_t i_altup_act  = 0; // altup_active_idx

    // granite-switch: the layer carrying adapter routing signal rather than positional info.
    // -1 means none. has_rope() below refuses to RoPE-shift it.
    int32_t router_layer = -1;

    // hunyuan-vl: NTK-aware alpha for XDRoPE
    float rope_scaling_alpha = 0.0f;

    // mamba: whether dt/B/C are RMS-normed. NOTE: this is NOT this fork's ssm_mamba2 -- that one
    // reads no GGUF key at all and is set by inference at llama-hparams.cpp:1626, while this is
    // read straight from LLM_KV_SSM_DT_B_C_RMS at src/models/mamba.cpp:8.
    bool ssm_dt_b_c_rms = false;

    // inkling (S69 P4). Names and defaults as in the saved upstream src__llama-hparams.h:93-99 (MIT). The short-convolution
    // kernel length is the existing n_shortconv_l_cache and the logit scale the existing f_logit_scale (= 1/logit_scale_denom).
    // Not added in P4: upstream's n_embd_r_impl (the packed 4-stream conv-state width) -- that belongs to the recurrent-memory
    // step (P2), where its consumers are.
    uint32_t inkling_d_rel            = 0;
    uint32_t inkling_rel_extent       = 0;     // global (non-SWA) layers
    uint32_t inkling_rel_extent_swa   = 0;     // local (SWA) layers
    uint32_t inkling_log_n_floor      = 0;     // 0 = no log-N logit scaling
    float    inkling_log_alpha        = 0.0f;
    uint32_t inkling_unpadded_n_vocab = 0;     // 0 = no padded-vocab masking

    // upstream's per-layer maximum; this fork keeps ONE n_expert_used for every layer, so the maximum is that value
    // upstream n_expert_used(il) (merge 2026-11-04): the per-layer array; this fork's loaders set only the scalar, so it
    // answers for a layer whose array entry is 0. n_expert_used_max(): upstream's max over the layers, plus the scalar.
    uint32_t n_expert_used_at(uint32_t il = 0) const {
        GGML_ASSERT(il < n_layer && "n_expert_used_at: layer out of range");
        return n_expert_used_arr[il] ? n_expert_used_arr[il] : n_expert_used;
    }
    uint32_t n_expert_used_max() const {
        uint32_t v = n_expert_used;
        for (uint32_t il = 0; il < n_layer && il < LLAMA_MAX_LAYERS; ++il) {
            v = n_expert_used_arr[il] > v ? n_expert_used_arr[il] : v;
        }
        return v;
    }

    // ---- methods adopted with the fields above; bodies in llama-hparams.cpp ----
    bool     is_mla() const;
    uint32_t n_embd_head_k_mla() const;
    uint32_t n_embd_head_v_mla() const;
    bool     has_rope(uint32_t il) const;
    uint32_t n_embd_inp_enc() const;
    bool     use_mrope() const;
    // ================================ end adopted block =============================

    bool operator!=(const llama_hparams & other) const {
        if (this->vocab_only    != other.vocab_only)    return true;
        if (this->n_vocab       != other.n_vocab)       return true;
        if (this->n_ctx_train   != other.n_ctx_train)   return true;
        if (this->n_embd        != other.n_embd)        return true;
        if (this->mtp_backbone_n_embd != other.mtp_backbone_n_embd) return true;
        if (this->dflash_block_size != other.dflash_block_size) return true;
        if (this->dflash_mask_token_id != other.dflash_mask_token_id) return true;
        if (this->dflash_n_target_features != other.dflash_n_target_features) return true;
        if (this->dflash_n_target_layers != other.dflash_n_target_layers) return true;
        if (this->dflash_conv_kernel_size != other.dflash_conv_kernel_size) return true;
        if (this->dflash_conv_group_size != other.dflash_conv_group_size) return true;
        if (this->dflash_selector_rank != other.dflash_selector_rank) return true;
        if (this->dflash_selector_top_k != other.dflash_selector_top_k) return true;
        if (this->dflash_laguna != other.dflash_laguna) return true;
        if (this->dflash_dsv4   != other.dflash_dsv4)   return true;
        if (this->dflash_dsv41  != other.dflash_dsv41)  return true;
        if (this->dflash_block_bidir != other.dflash_block_bidir) return true;
        if (this->n_layer       != other.n_layer)       return true;
        if (this->n_rot         != other.n_rot)         return true;
        if (this->n_swa         != other.n_swa)         return true;
        // FIXED 2026-09-09, slice 31. This read `return false;` -- the only comparison in this
        // operator that did, against more than a hundred siblings returning true. Two effects, the
        // second worse: hparams differing only in n_swa_pattern reported NOT-unequal, i.e. EQUAL;
        // and the early return SHORT-CIRCUITED the 51 comparisons below it, so a difference here
        // made two otherwise entirely different hparams compare equal regardless of everything
        // else. No reading makes that correct -- deliberately ignoring a field is written by
        // omitting its comparison, not by a return that also discards 51 later tests.
        // Latent, not live: this operator had ZERO callers when the fix was made (measured over
        // src/ with word boundaries; the only other hit was an unrelated probs_iterator::operator!=
        // in llama-sampler.cpp). Fixed anyway, because upstream uses hparams equality to check that
        // a loaded session matches the current model, and that is precisely the check whose silent
        // failure is dangerous.
        if (this->n_swa_pattern != other.n_swa_pattern) return true;
        if (this->n_embd_head_k_full != other.n_embd_head_k_full) return true;
        if (this->n_embd_head_v_full != other.n_embd_head_v_full) return true;
        if (this->n_expert      != other.n_expert)      return true;
        if (this->n_expert_used != other.n_expert_used) return true;

        if (this->n_head_arr    != other.n_head_arr)    return true;
        if (this->n_head_kv_arr != other.n_head_kv_arr) return true;
        if (this->n_ff_arr      != other.n_ff_arr)      return true;

        if (this->n_rel_attn_bkts    != other.n_rel_attn_bkts)    return true;
        if (this->n_layer_dense_lead != other.n_layer_dense_lead) return true;
        if (this->n_lora_q           != other.n_lora_q)           return true;
        if (this->n_lora_kv          != other.n_lora_kv)          return true;
        if (this->n_ff_exp           != other.n_ff_exp)           return true;
        if (this->n_ff_shexp         != other.n_ff_shexp)         return true;
        if (this->n_expert_shared    != other.n_expert_shared)    return true;
        if (this->n_value_expert      != other.n_value_expert)      return true;
        if (this->n_value_expert_used != other.n_value_expert_used) return true;

        if (this->rope_finetuned  != other.rope_finetuned)  return true;
        if (this->n_ctx_orig_yarn != other.n_ctx_orig_yarn) return true;

        if (this->ssm_d_conv  != other.ssm_d_conv)  return true;
        if (this->ssm_d_inner != other.ssm_d_inner) return true;
        if (this->ssm_d_state != other.ssm_d_state) return true;
        if (this->ssm_dt_rank != other.ssm_dt_rank) return true;
        if (this->ssm_n_group != other.ssm_n_group) return true;
        if (this->kda_safe_gate != other.kda_safe_gate) return true;
        if (this->ttt_layer_impl   != other.ttt_layer_impl)   return true;
        if (this->ttt_chunk        != other.ttt_chunk)        return true;
        if (this->ttt_lr           != other.ttt_lr)           return true;
        if (this->ttt_target       != other.ttt_target)       return true;
        if (this->ttt_conv_padding != other.ttt_conv_padding) return true;
        if (this->ttt_clip_tau     != other.ttt_clip_tau)     return true;
        if (this->is_recr_impl != other.is_recr_impl) return true;
        if (this->n_shortconv_l_cache != other.n_shortconv_l_cache) return true;
        if (this->inkling_d_rel            != other.inkling_d_rel)            return true;
        if (this->inkling_rel_extent       != other.inkling_rel_extent)       return true;
        if (this->inkling_rel_extent_swa   != other.inkling_rel_extent_swa)   return true;
        if (this->inkling_log_n_floor      != other.inkling_log_n_floor)      return true;
        if (this->inkling_unpadded_n_vocab != other.inkling_unpadded_n_vocab) return true;
        for (int i = 0; i < 8; ++i) {
            if (this->dflash_target_layer_ids[i] != other.dflash_target_layer_ids[i]) return true;
        }

        if (this->dec_start_token_id != other.dec_start_token_id) return true;

        const float EPSILON = 1e-9f;

        if (!is_float_close(this->f_norm_eps,            other.f_norm_eps,            EPSILON)) return true;
        if (!is_float_close(this->f_norm_rms_eps,        other.f_norm_rms_eps,        EPSILON)) return true;
        if (!is_float_close(this->kda_gate_lower_bound,  other.kda_gate_lower_bound,  EPSILON)) return true;
        if (!is_float_close(this->rope_attn_factor,      other.rope_attn_factor,      EPSILON)) return true;
        if (!is_float_close(this->rope_freq_base_train,  other.rope_freq_base_train,  EPSILON)) return true;
        if (!is_float_close(this->rope_freq_scale_train, other.rope_freq_scale_train, EPSILON)) return true;
        if (!is_float_close(this->expert_weights_scale,  other.expert_weights_scale,  EPSILON)) return true;
        if (!is_float_close(this->rope_yarn_log_mul,     other.rope_yarn_log_mul,     EPSILON)) return true;
        if (!is_float_close(this->rope_yarn_mscale,      other.rope_yarn_mscale,      EPSILON)) return true;
        if (this->rope_yarn_truncate != other.rope_yarn_truncate) return true;
        if (!is_float_close(this->f_residual_scale,      other.f_residual_scale,      EPSILON)) return true;
        if (!is_float_close(this->f_embedding_scale,     other.f_embedding_scale,     EPSILON)) return true;
        if (!is_float_close(this->f_attention_scale,     other.f_attention_scale,     EPSILON)) return true;
        if (!is_float_close(this->f_attn_v_scale,        other.f_attn_v_scale,        EPSILON)) return true;
        if (!is_float_close(this->inkling_log_alpha,     other.inkling_log_alpha,     EPSILON)) return true;
        if (!is_float_close(this->f_logit_scale,         other.f_logit_scale,         EPSILON)) return true;

        return false;
    }

    bool has_kv(uint32_t il) const {
        return n_layer_kv_from_start > 0 ? il < n_layer_kv_from_start : true;
    }

    uint32_t n_head(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_arr[il];
        }
        printf("%s: Oops, il = %d\n", __func__, il);
        GGML_ABORT("fatal error");
    }

    uint32_t n_head_kv(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_kv_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_embd_inp() const {
        // ADDED 2026-09-08 with the n_embd_inp_impl field, from
        // 279b4bfaf^2:src/llama-hparams.cpp:93-96. This early return is NOT optional decoration:
        // GGUF loaders write n_embd_inp_impl, and without this check the field would be written
        // and never read -- a silent no-op, which is precisely the defect class the adoption of
        // these members exists to fix. The deepstack behaviour below is this fork's and is
        // preserved unchanged for the (normal) case where the field is unset.
        if (n_embd_inp_impl > 0) {
            return n_embd_inp_impl;
        }

        uint32_t n_embd_inp = n_embd;

        if (n_deepstack_layers > 0) {
            n_embd_inp += n_embd * n_deepstack_layers;
        }

        return n_embd_inp;
    }

    uint32_t n_ff(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_ff_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    // hy_v4 (slice 47d). Upstream's n_ff_exp is an ACCESSOR over
    // n_ff_exp_arr; this fork already uses that NAME for a plain scalar
    // field, so the accessor is added under a distinct name rather than
    // shadowing or renaming a field 20+ architectures already read.
    // The array is the correct source for hy_v4 specifically:
    // src/models/hy-v4.cpp's own load_arch_hparams fills n_ff_exp_arr via
    // get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, ...) and never
    // writes the scalar, so reading the scalar would read a value this
    // architecture never sets.
    uint32_t n_ff_exp_at(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_ff_exp_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_gqa(uint32_t il = 0) const {
        const uint32_t n_head    = this->n_head(il);
        const uint32_t n_head_kv = this->n_head_kv(il);

        if (n_head_kv == 0) {
            return 0;
        }

        return n_head/n_head_kv;
    }

    // DEFAULT ARGUMENT ADDED 2026-09-08. Upstream declares both of these as
    // `n_embd_head_k(uint32_t il = 0)`, and the merged src/llama-graph.cpp:1464,1466 calls them
    // with no argument:
    //     error: no matching function for call to 'llama_hparams::n_embd_head_k() const'
    // Adding the default is the whole fix and it changes nothing for the 40+ existing callers that
    // pass a layer. Layer 0 is upstream's own meaning for the no-argument form, so the two agree.
    uint32_t n_embd_head_k(int il = 0) const { return is_swa_impl[il] ? n_embd_head_k_swa : n_embd_head_k_full; }

    uint32_t n_embd_head_v(int il = 0) const { return is_swa_impl[il] ? n_embd_head_v_swa : n_embd_head_v_full; }

    // ------------------------------------------------------------------------------------------
    // ADDED 2026-09-08 -- upstream accessors the merge dropped while keeping their callers.
    //
    // n_layer_eff() is upstream's `n_layer()`, which returns the EFFECTIVE layer count with the
    // nextn/MTP layers excluded (279b4bfaf^2:src/llama-hparams.cpp:301, `n_layer_all -
    // n_layer_nextn`). It CANNOT be called n_layer() here: this struct already has a FIELD of that
    // name, and a struct has one member per name. That is not a cosmetic clash -- upstream's
    // n_layer() and this fork's n_layer field mean DIFFERENT NUMBERS whenever a model has nextn
    // layers, so a merged file calling hparams.n_layer() and getting the field back would silently
    // read the total where it asked for the effective count. Renaming the accessor makes the two
    // impossible to confuse; the merged call sites are rewritten to it.
    //
    // This fork spells the nextn count `nextn_predict_layers` rather than upstream's
    // `n_layer_nextn`, and BOTH fields now exist in this struct (n_layer_nextn was adopted earlier
    // in this merge). They are the same quantity under two names and will drift. The subtraction
    // below deliberately uses the fork's field, because the fork's loaders are what populate it --
    // reconciling the two is real follow-up, recorded, not silently papered over here.
    uint32_t n_layer_eff() const {
        return n_layer > nextn_predict_layers ? n_layer - nextn_predict_layers : n_layer;
    }

    // Positions consumed per embedding. Adopted verbatim from
    // 279b4bfaf^2:src/llama-hparams.cpp:239 -- M-RoPE and interleaved M-RoPE carry four position
    // components per token (t, h, w, and the section index), everything else carries one.
    uint32_t n_pos_per_embd() const {
        return rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
    }

    // n_embd_r() / n_embd_s() are upstream's names for the recurrent-state sizes this fork calls
    // n_embd_k_s() / n_embd_v_s(). They are NOT a pure rename: the two implementations have
    // genuinely diverged and cover DIFFERENT ARCHITECTURE FAMILIES, which is why this is a merge
    // and not an alias.
    //
    //   this fork covers   mamba2 / granitehybrid (ssm_mamba2), qwen3next (ssm_n_group), and
    //                      plain Mamba. Its qwen3next branch computes a delta-net state that
    //                      upstream's version has no equivalent for.
    //   upstream covers    RWKV (wkv_head_size), LFM2 (n_shortconv_l_cache), Kimi KDA
    //                      (n_embd_head_kda) and MiniMax-Text-01 linear attention (n_embd_head_la).
    //
    // So the fork's branches are tried FIRST and unchanged -- they are the ones this tree's loaders
    // actually populate and the ones its models are tested against -- and upstream's branches are
    // added for the families the fork's version does not handle. The two fallbacks agree where they
    // meet: upstream's plain-Mamba conv size is (d_conv-1)*(d_inner + 2*n_group*d_state) and the
    // fork's is (d_conv-1)*d_inner, and the fork's fallback is only reached when ssm_n_group == 0,
    // where the extra term is zero.
    //
    // The four upstream gate fields (wkv_head_size, n_embd_head_kda, n_embd_head_la, and
    // token_shift_count) are declared above with upstream's defaults. Until a loader sets them --
    // those loaders live in src/models/, which is not yet built -- every upstream branch below is
    // inert and these functions return exactly what n_embd_k_s()/n_embd_v_s() returned before.
    // That is deliberate: adding the branch now means the day src/models/ is wired, the RWKV and
    // KDA paths are already correct rather than silently returning a Mamba-shaped size.
    uint32_t n_embd_r() const {
        if (wkv_head_size != 0) {
            // RWKV: the token-shift buffer, not a conv state
            return token_shift_count * n_embd;
        }
        if (n_shortconv_l_cache != 0) {
            // LFM2 short convolution
            return n_embd * (n_shortconv_l_cache - 1);
        }
        if (n_embd_head_kda != 0) {
            // Kimi KDA: conv state for Q, K and V -- 3 * (d_conv - 1) * n_head * head_dim
            const uint32_t d_inner = n_head() * n_embd_head_kda;
            return 3 * (ssm_d_conv > 0 ? ssm_d_conv - 1 : 3) * d_inner;
        }
        return n_embd_k_s();
    }

    uint32_t n_embd_s() const {
        if (wkv_head_size != 0) {
            // RWKV's wkv_states size
            return n_embd * wkv_head_size;
        }
        if (n_embd_head_kda != 0) {
            // Kimi KDA: the full recurrent state, [head_dim, head_dim, n_head]
            return n_embd_head_kda * n_embd_head_kda * n_head();
        }
        if (n_embd_head_la != 0) {
            // MiniMax-Text-01 linear attention, same shape
            return n_embd_head_la * n_embd_head_la * n_head();
        }
        return n_embd_v_s();
    }

    uint32_t n_embd_k_gqa(uint32_t il = 0) const { // dimension of key embeddings across all k-v heads
        return n_head_kv(il) * n_embd_head_k(il);
    }

    uint32_t n_embd_v_gqa(uint32_t il = 0) const { // dimension of value embeddings across all k-v heads
        return n_head_kv(il) * n_embd_head_v(il);
    }

    uint32_t n_embd_k_s() const { // dimension of the rolling state embeddings
        if (ssm_split_state) {
            return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * (ssm_d_inner + 2 * ssm_n_group * ssm_d_state);
        }
        if (ssm_mamba2) {
            // granitehybrid keeps conv AND ssm state together in the s_l tail
            return 0;
        }
        if (ssm_n_group > 0) {
            // qwen3next keeps all recurrent state in the V-cache tail
            return 0;
        }
        // corresponds to Mamba's conv_states size
        // TODO: maybe support other convolution strides than 1
        // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
        return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * ssm_d_inner;
    }

    uint32_t n_embd_v_s() const { // dimension of the recurrent state embeddings
        if (ssm_split_state) {
            return ssm_d_state * ssm_d_inner;
        }
        if (ssm_mamba2) {
            // mamba2 (granitehybrid) packs, per recurrent layer:
            //   1) conv state: (d_conv - 1) * (d_inner + 2 * n_group * d_state)
            //      -- the conv runs over x AND over B and C, not over x alone
            //   2) ssm state:  d_state * d_inner
            const uint32_t conv_dim = ssm_d_inner + 2 * ssm_n_group * ssm_d_state;
            const uint32_t conv_state_dim = (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * conv_dim;
            return conv_state_dim + ssm_d_state * ssm_d_inner;
        }
        if (ssm_n_group > 0) {
            // qwen3next recurrent state packs:
            // 1) conv state: (d_conv - 1) * (2 * key_dim + value_dim)
            // 2) delta-net state: head_v_dim * head_v_dim * num_v_heads
            const uint32_t key_dim        = ssm_d_state * ssm_n_group;
            const uint32_t value_dim      = ssm_d_inner;
            const uint32_t conv_dim       = 2 * key_dim + value_dim;
            const uint32_t conv_state_dim = (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * conv_dim;
            const uint32_t head_v_dim     = ssm_dt_rank > 0 ? ssm_d_inner / ssm_dt_rank : 0;
            const uint32_t ssm_state_dim  = head_v_dim * head_v_dim * ssm_dt_rank;
            return conv_state_dim + ssm_state_dim;
        }
        // corresponds to Mamba's ssm_states size
        return ssm_d_state * ssm_d_inner;
    }

    std::pair<uint32_t, uint32_t> n_embd_v_s_dims(int nv) const {
        if (ssm_n_group <= 0 || nv < 1 || ssm_dt_rank < 1) return {0, 0};
        int num_v_heads = ssm_dt_rank;
        int num_k_heads = ssm_n_group;
        int gqa_ratio   = num_v_heads / num_k_heads;
        GGML_ASSERT(nv <= num_v_heads);
        GGML_ASSERT(nv % gqa_ratio == 0);
        int nk = nv / gqa_ratio;
        int head_k_dim  = ssm_d_state;
        int head_v_dim  = ssm_d_inner / num_v_heads;
        uint32_t conv_dim       = 2 * nk * head_k_dim + nv * head_v_dim;
        //uint32_t conv_state_dim = conv_dim * (ssm_d_conv - 1);
        //uint32_t ssm_state_dim  = head_v_dim * head_v_dim * nv;
        //return {conv_state_dim, ssm_state_dim};
        uint32_t ssm_state_dim  = head_v_dim * head_v_dim * nv;
        return {conv_dim, ssm_state_dim};
    }

    uint32_t n_embd_v_s_id(int nv) const {
        auto [conv_dim, ssm_state_dim] = n_embd_v_s_dims(nv);
        return (ssm_d_conv - 1) * conv_dim + ssm_state_dim;
    }

    bool is_recurrent(uint32_t il) const {
        return il < n_layer ? is_recr_impl[il] : false;
    }

    // ------------------------------------------------------------------------------------------
    // ADDED 2026-09-08 -- upstream's SWA / recurrent / indexer accessors, which the merge dropped
    // while keeping all of their callers. Reference counts measured over src/, tools/ and common/:
    //     is_swa           73     set_swa_pattern  22     is_masked_swa     6
    //     is_recr          27     is_indexer_full   2     is_swa_any        0 (kept anyway)
    // Every one of them was calling a member that did not exist.
    //
    // THESE ARE VIEWS OVER THE EXISTING ARRAYS, NOT NEW STATE. That is the whole point. The
    // obvious alternative -- give upstream's API its own storage -- puts two arrays in this struct
    // holding the same fact, and they drift the instant one loader writes only one of them: the
    // fork's loaders populate the per-layer array at 15 sites and src/models/'s at 14. That
    // failure is SILENT. The model loads, SWA is simply not applied, and the only symptom is
    // slightly worse output. Nothing in this tree would catch it. So the arrays were renamed to
    // upstream's spellings (is_swa_impl / is_recr_impl / is_indexer_full_impl, 86 sites, commit
    // this same day) and these accessors read them.
    //
    // OUT-OF-RANGE BEHAVIOUR follows the FORK's existing convention -- return false, as
    // is_recurrent() directly above has always done -- rather than upstream's GGML_ABORT. Two
    // accessors over one array that disagree about the edge case is a worse bug than either
    // choice: a caller would get different answers for il >= n_layer depending on which spelling
    // it happened to use.
    bool is_swa(uint32_t il) const {
        return il < n_layer ? is_swa_impl[il] != 0 : false;
    }

    // true if ANY layer is SWA. Kept although nothing in this tree calls it yet: it is part of
    // upstream's public shape, the next merge's src/models/ will use it, and it cannot be wrong.
    bool is_swa_any() const {
        for (uint32_t il = 0; il < n_layer && il < LLAMA_MAX_LAYERS; ++il) {
            if (is_swa_impl[il]) {
                return true;
            }
        }
        return false;
    }

    bool is_recr(uint32_t il) const { return is_recurrent(il); }

    bool is_indexer_full(uint32_t il) const {
        return il < n_layer ? is_indexer_full_impl[il] : false;
    }

    // Mark every n_pattern-th layer. Adopted from 279b4bfaf^2:src/llama-hparams.cpp:8-22 with one
    // deliberate difference: upstream clears layers between n_layer() and n_layer_all, a split
    // this fork does not have (it carries a single n_layer plus nextn_predict_layers), so the
    // clear runs to LLAMA_MAX_LAYERS instead. Same effect -- no stale layer is left marked -- over
    // the range this struct actually addresses.
    //
    //   n_pattern = 0     every layer SWA
    //   n_pattern = 1     no layer SWA (il % 1 < 0 is never true)
    //   n_pattern = N, dense_first = false   layers 0..N-2 SWA, layer N-1 dense, repeating
    //   n_pattern = N, dense_first = true    layer 0 dense, 1..N-1 SWA, repeating
    void set_swa_pattern(uint32_t n_pattern, bool dense_first = false) {
        const uint32_t n = n_layer < LLAMA_MAX_LAYERS ? n_layer : LLAMA_MAX_LAYERS;
        for (uint32_t il = 0; il < n; ++il) {
            is_swa_impl[il] = dense_first
                ? (n_pattern == 0 || (il % n_pattern != 0))
                : (n_pattern == 0 || (il % n_pattern < (n_pattern - 1)));
        }
        for (uint32_t il = n; il < LLAMA_MAX_LAYERS; ++il) {
            is_swa_impl[il] = 0;
        }
    }

    // The recurrent twin of set_swa_pattern, same shape, same reasoning
    // (279b4bfaf^2:src/llama-hparams.cpp:24-38).
    void set_recr_pattern(uint32_t n_pattern, bool dense_first = false) {
        const uint32_t n = n_layer < LLAMA_MAX_LAYERS ? n_layer : LLAMA_MAX_LAYERS;
        for (uint32_t il = 0; il < n; ++il) {
            is_recr_impl[il] = dense_first
                ? (n_pattern == 0 || (il % n_pattern != 0))
                : (n_pattern == 0 || (il % n_pattern < (n_pattern - 1)));
        }
        for (uint32_t il = n; il < LLAMA_MAX_LAYERS; ++il) {
            is_recr_impl[il] = false;
        }
    }

    // Adopted VERBATIM from 279b4bfaf^2:src/llama-hparams.h:408-440, arithmetic unchanged.
    //
    // It takes n_swa and swa_type as ARGUMENTS rather than reading the members, and that is not an
    // accident of style: the kv-cache classes call it with their OWN window, which can differ from
    // hparams' -- see src/llama-kv-cache.cpp:1548, which pulls swa_type out of its args struct.
    // Keeping it static and parameterised is what lets one implementation serve both.
    //
    // Inlined here rather than in the .cpp for the reason upstream gives: it is called once per
    // (query, key) pair while building a mask, so a call per element would show up in the profile.
    static bool is_masked_swa(uint32_t n_swa, llama_swa_type swa_type, llama_pos p0, llama_pos p1) {
        GGML_ASSERT(p0 >= 0 && p1 >= 0);

        switch (swa_type) {
            case LLAMA_SWA_TYPE_NONE:
                {
                } break;
            case LLAMA_SWA_TYPE_STANDARD:
                {
                    if (p1 - p0 >= (int32_t) n_swa) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_CHUNKED:
                {
                    const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;

                    if (p0 < pos_chunk_start) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_SYMMETRIC:
                {
                    const int32_t half_n_swa = (int32_t) n_swa / 2;
                    const int32_t pos_diff   = p1 - p0;

                    // Mask if outside the symmetric window
                    if (pos_diff < -half_n_swa || pos_diff > half_n_swa) {
                        return true;
                    }
                } break;
        }

        return false;
    }

    // fork loaders mark PLE layers in ple_layer_arr; upstream's src/models classes (deepseek41.cpp:116, qwen4exp.cpp:103)
    // mark is_ple_impl -- either marks the layer (merge 2026-11-04)
    bool is_ple(uint32_t il) const {
        return il < n_layer ? (ple_layer_arr[il] || is_ple_impl[il]) : false;
    }

    // UPSTREAM's ple_conv_state(): the width of one PLE conv-history row of the separate cache_ple_r_l tensor
    // (llama-memory-recurrent.cpp). The fork's ple_conv_state() below is the history LENGTH in positions, read by
    // n_embd_ple_conv() and graphs/build_qwen4exp.cpp; upstream folded the dsv4_hc_mult * n_embd product into it.
    uint32_t ple_conv_row_width() const {
        return ple_n_heads == 0 ? 0 : ple_conv_state() * dsv4_hc_mult * n_embd;
    }

    // the layer runs Qwen sparse attention over pooled blocks; deepseek4 fills the same
    // ratio array for its CSA/HCA layers and reads the ratio value directly instead
    bool is_qsa(uint32_t il) const {
        return il < n_layer ? dsv4_compress_ratios[il] > 0 : false;
    }

    // rows the PLE convolution history adds to a layer's recurrent state row: the taps reach
    // (kernel - 1) * ngram_size positions back, over every channel of the wide residual
    uint32_t n_embd_ple_conv(uint32_t il) const {
        return is_ple(il) ? ple_conv_state() * dsv4_hc_mult * n_embd : 0;
    }

    uint32_t ple_conv_state() const {
        return ple_conv_kernel > 0 ? (ple_conv_kernel - 1) * ple_ngram_size : 0;
    }

    // Inkling (S69 P1): every layer is an attention layer AND carries four short convolutions (key, value, attention
    // output, mlp output), each keeping its last (kernel - 1) inputs. The four histories are packed into one F32 state
    // row per sequence, in the order [k | v | attn | mlp]; each stream is stored channel-major ([kernel - 1, width]),
    // which is the layout ggml_ssm_conv reads. The key and value widths differ between global and sliding-window
    // layers, so the row width is per layer. inkling_d_rel is the arch's own marker: nothing else sets it.
    uint32_t n_embd_inkling_conv(uint32_t il) const {
        if (inkling_d_rel == 0 || n_shortconv_l_cache <= 1 || il >= n_layer) {
            return 0;
        }
        return (n_shortconv_l_cache - 1) * (n_embd_k_gqa(il) + n_embd_v_gqa(il) + 2*n_embd);
    }

    // everything an ATTENTION layer keeps in its extra state row: the PLE history, or Inkling's four conv histories
    uint32_t n_embd_attn_layer_state(uint32_t il) const {
        return n_embd_ple_conv(il) + n_embd_inkling_conv(il);
    }

    struct recurrent_state_layout {
        uint32_t conv_width;
        uint32_t conv_feature_width;
        uint32_t ssm_width;
        uint32_t ple_offset;
        uint32_t ple_width;
        uint32_t row_width;
        bool has_ple;
    };

    recurrent_state_layout recurrent_state_layout_for(uint32_t il) const {
        const auto [conv_dim, ssm_width] = n_embd_v_s_dims(ssm_dt_rank);
        const uint32_t conv_width = (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * conv_dim;
        const uint32_t ple_width = n_embd_ple_conv(il);
        // an ATTENTION layer with a PLE history has a state row holding ONLY the history (llama.cpp:1981), so the
        // history starts at 0 there; conv+ssm is the offset only on a recurrent layer (S69-ENGREV-...-PLE-STATE-TWO-LAYOUTS)
        const uint32_t ple_offset = is_recurrent(il) ? conv_width + ssm_width : 0;
        return {
            conv_width,
            conv_dim,
            ssm_width,
            ple_offset,
            ple_width,
            ple_offset + ple_width,
            ple_width > 0,
        };
    }

    static bool is_float_close(float a, float b, float abs_tol) {
        // Check for non-negative tolerance
        if (abs_tol < 0.0) {
            throw std::invalid_argument("Tolerance must be non-negative");
        }

        // Exact equality check
        if (a == b) {
            return true;
        }

        // Check for infinities
        if (std::isinf(a) || std::isinf(b)) {
            return false;
        }

        // Regular comparison using the provided absolute tolerance
        return std::fabs(b - a) <= abs_tol;
    }

    // ADDED 2026-09-09. upstream's n_embd_out() is NOT this fork's n_embd_out field, and the
    // difference is a real defect rather than a spelling:
    //     upstream  (279b4bfaf^2:src/llama-hparams.cpp:111)
    //         uint32_t llama_hparams::n_embd_out() const {
    //             return n_embd_out_impl > 0 ? n_embd_out_impl : n_embd;
    //         }
    //     this fork   uint32_t n_embd_out = 0;      -- the field, with NO fallback anywhere
    // The merge adopted the FIELD and left the FALLBACK behind. Any model that does not set
    // LLM_KV_EMBEDDING_LENGTH_OUT leaves it 0, and the merged call sites size TENSORS with it:
    // src/models/lfm2.cpp and lfm2moe.cpp build LLM_TENSOR_DENSE_2_OUT as {n_embd, n_embd_out()}
    // with a bias of {n_embd_out()}, and src/models/wavtokenizer-dec.cpp builds LLM_TENSOR_OUTPUT
    // the same way. A zero dimension there is a broken tensor, not merely a wrong number -- so
    // rewriting those calls to the bare field would have been worse than leaving them uncompiled.
    //
    // It cannot be named n_embd_out() for the same reason n_layer_eff() cannot be named n_layer():
    // the field already owns that name in this struct, and a struct has one member per name. The
    // _eff suffix follows the precedent set at n_layer_eff() rather than inventing a second
    // convention.
    uint32_t n_embd_out_eff() const {
        return n_embd_out > 0 ? n_embd_out : n_embd;
    }

    uint32_t rope_n_rot(uint32_t il) const {
        const uint32_t v = rope_dim_per_layer[il];
        return v ? v : is_swa_impl[il] ? n_rot_swa : n_rot;
    }

    static bool is_v41_separate();

    static const char * rope_scaling_type_name(llama_rope_scaling_type);

};

static_assert(std::is_trivially_copyable<llama_hparams>::value, "llama_hparams must be trivially copyable");

// sinks + retained window + one u-batch, padded as one sum so the total is pad-aligned for any
// sink_rows; compaction then fires every C - W tokens, and the slack floor keeps a small u-batch from compacting every few tokens
static inline uint32_t llama_swa_compact_rows(uint32_t window, uint32_t pad, uint32_t n_ubatch,
                                              uint32_t sink_rows) {
    const uint32_t min_slack = 256;
    const uint32_t slack     = n_ubatch > min_slack ? n_ubatch : min_slack;
    const uint32_t unpadded  = sink_rows + window + slack;
    return pad > 1 ? ((unpadded + pad - 1)/pad)*pad : unpadded;
}

static inline uint32_t llama_kv_layer_rows(const llama_hparams & hparams, int il, uint32_t kv_size,
                                           bool swa_compress, uint32_t n_ubatch, uint32_t pad) {
    if (!swa_compress || il < 0 || il >= (int) hparams.n_layer) {
        return kv_size;
    }
    if (il >= (int) (hparams.n_layer - hparams.nextn_predict_layers)) {
        return kv_size;
    }
    if (!hparams.is_swa_impl[il]) {
        return kv_size;
    }
    const uint32_t rows = llama_swa_compact_rows(hparams.n_swa, pad, n_ubatch, hparams.param_sink_number);
    return rows < kv_size ? rows : kv_size;
}
