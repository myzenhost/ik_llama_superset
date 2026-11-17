#pragma once

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-hparams.h"
#include "llama.h"
// adopted from upstream in this merge -- new dependencies this file did not have before, needed
// for the real llama_memory_t implementation and its graph-building types.
#include "llama-graph.h"
#include "llama-memory.h"
// adopted from upstream in this merge (S69-SRC-MODELS-BLOCKED-BY-MISSING-LLAMA-MODEL-BASE-
// HIERARCHY-2026-09-08-01): llama_model_base (bottom of this file) needs the FULL definition of
// llama_model_loader (its TENSOR_* constants and create_tensor(ctx, name, ne, flags)), not just
// the forward declaration below. Safe: llama-model-loader.h does not include this file or
// "models.h" (checked directly, no cycle) -- confirmed by this file compiling both before and
// after this include was added.
#include "llama-model-loader.h"

#include "ggml-backend.h"

#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <memory>
#include <string>

#include "llama-reload-info.h"

struct llama_cparams;
struct llama_ubatch;
struct llama_model_loader;
struct llama_model;

// Adopted from upstream in this merge. The callers are real -- src/llama-model-saver.cpp:320 and
// src/models/modern-bert.cpp:22 -- but the second half of the note that stood here until 2026-09-09
// was FALSE: it said "defined in llama-model.cpp", and neither symbol was defined anywhere in the
// tree. A comment asserting a definition is not a definition, and nothing checked it for two weeks.
// Both are now defined in src/llama-hparams.cpp, beside the tables they read.
std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type);
std::string llama_rope_section_order_name(llama_rope_section_order rope_section_order);

// Map a GGUF activation-name string to llm_ffn_op_type. Returns `fallback` if
// the string is empty or not recognized.
llm_ffn_op_type llm_ffn_op_type_from_string(const std::string & name, llm_ffn_op_type fallback);

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_14M,
    MODEL_17M,
    MODEL_22M,
    MODEL_33M,
    MODEL_60M,
    MODEL_70M,
    MODEL_80M,
    MODEL_109M,
    MODEL_137M,
    MODEL_140M,
    MODEL_160M,
    MODEL_190M,
    MODEL_220M,
    MODEL_250M,
    MODEL_256M,
    MODEL_270M,
    MODEL_335M,
    MODEL_350M,
    MODEL_360M,
    MODEL_410M,
    MODEL_450M,
    MODEL_475M,
    MODEL_558M,
    MODEL_700M,
    MODEL_770M,
    MODEL_780M,
    MODEL_950M,
    MODEL_0_3B,
    MODEL_0_5B,
    MODEL_0_6B,
    MODEL_0_8B,
    MODEL_1B,
    MODEL_1_2B,
    MODEL_1_3B,
    MODEL_1_4B,
    MODEL_1_5B,
    MODEL_1_6B,
    MODEL_1_7B,
    MODEL_1_8B,
    MODEL_2B,
    MODEL_2_6B,
    MODEL_2_8B,
    MODEL_2_9B,
    MODEL_3B,
    MODEL_4B,
    MODEL_6B,
    MODEL_6_9B,
    MODEL_7B,
    MODEL_8B,
    MODEL_9B,
    MODEL_11B,
    MODEL_12B,
    MODEL_13B,
    MODEL_14B,
    MODEL_15B,
    MODEL_16B,
    MODEL_20B,
    MODEL_26B_A4B,
    MODEL_27B,
    MODEL_30B,
    MODEL_32B,
    MODEL_34B,
    MODEL_35B,
    MODEL_36B,
    MODEL_36B_A4B, // K2-Horizon MoVA
    MODEL_40B,
    MODEL_65B,
    MODEL_70B,
    MODEL_120B,
    MODEL_142B,
    MODEL_236B,
    MODEL_290B,
    MODEL_314B,
    MODEL_405B,
    MODEL_671B,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
    MODEL_XL,
    MODEL_A1_7B,
    MODEL_A2_7B,
    MODEL_8x7B,
    MODEL_8x22B,
    MODEL_16x12B,
    MODEL_16x3_8B,
    MODEL_10B_128x3_66B,
    MODEL_57B_A14B,
    MODEL_17B_16E,
    MODEL_17B_128E,
    MODEL_A13B,
    MODEL_7B_A1B,
    MODEL_8B_A1B,
    MODEL_12B_A2_5B,
    MODEL_16B_A1B,
    MODEL_21B_A3B, // Ernie MoE small
    MODEL_30B_A3B,
    MODEL_33B_A3B,
    MODEL_35B_A3B,
    MODEL_80B_A3B, // Qwen3-Next
    MODEL_125B_A6B, // Qwen3.8-Flash-Next
    MODEL_80B_A13B,
    MODEL_100B_A6B,
    MODEL_106B_A12B,
    MODEL_119B_A6B,
    MODEL_122B_A10B,
    MODEL_230B_A10B, // Minimax M2
    MODEL_235B_A22B,
    MODEL_310B_A15B,
    MODEL_312B_A17B,
    MODEL_300B_A47B, // Ernie MoE big
    MODEL_355B_A32B,
    MODEL_397B_A17B, // Qwen-3.5-MoE
    MODEL_744B_A40B,
    MODEL_E2B,
    MODEL_E4B,
    // adopted from upstream in this merge -- 20 additional model-size buckets this
    // enum lacked (DeepSeek V3.2, GLM-5, Kimi-K3, Minimax M3, Ling-3.0 variants,
    // Solar-Open, Laguna-S-2, Nemotron 3 Super, Step3.5, MiMo-V2, dots3, etc.), named in
    // this enum's own MODEL_ convention (upstream calls the enum llm_type / LLM_TYPE_*;
    // kept the fork's e_model/MODEL_ name since it is what the live path uses --
    // confirmed by src/llama-model.cpp's `case MODEL_7B:` inside the resolved,
    // non-conflicted region of this merge).
    MODEL_47M,
    MODEL_149M,
    MODEL_230M,
    MODEL_395M,
    MODEL_26B,
    MODEL_31B,
    MODEL_456B,
    MODEL_7_9B_A1_3B, // Ling-3.0-tiny
    MODEL_24B_A2B, // lfm2moe
    // upstream size buckets added 2026-11-04 (merge); alias defines below
    MODEL_816B_A42B,
    MODEL_78B_A3_5B,
    MODEL_1B_A400M,
    MODEL_3B_A800M,
    MODEL_32B_A9B,
    MODEL_75B_A9B,
    MODEL_A3B,
    MODEL_320B_A18B,
    MODEL_31B_A3_5B,
    MODEL_48B_A3B, // Kimi Linear
    MODEL_102B_A12B, // Solar-Open
    MODEL_118B_A8B, // Laguna-S-2
    MODEL_120B_A12B, // Nemotron 3 Super
    MODEL_124B_A5_1B, // Ling-3.0-flash
    MODEL_196B_A11B, // Step3.5-Flash
    MODEL_428B_A23B, // Minimax M3
    MODEL_288B_A19B, // dots3-note
    MODEL_685B_A37B, // DeepSeek V3.2
    MODEL_2_8T_A50B, // Kimi-K3
};

// adopted from upstream in this merge: strict superset of the fork's original 6-field struct
// (adds eh_proj_s/eh_proj_in_s/shared_head_head_s/shared_head_head_in_s -- per-tensor and
// per-input NVFP4-style scales for the same MTP head tensors). Confirmed needed: measured by
// diffing member names, fork's 6 are a proper subset of upstream's 10, so this is pure addition.
struct llama_layer_nextn {
    struct ggml_tensor * eh_proj               = nullptr;
    struct ggml_tensor * eh_proj_s             = nullptr;
    struct ggml_tensor * eh_proj_in_s          = nullptr;
    struct ggml_tensor * embed_tokens          = nullptr;
    struct ggml_tensor * enorm                 = nullptr;
    struct ggml_tensor * hnorm                 = nullptr;
    struct ggml_tensor * shared_head_head      = nullptr;
    struct ggml_tensor * shared_head_head_s    = nullptr;
    struct ggml_tensor * shared_head_head_in_s = nullptr;
    struct ggml_tensor * shared_head_norm      = nullptr;

    // qwen4exp: the NextN head's final hyper-connection mixer. For merged
    // community files these are resolved to shared_head_norm + the trunk mixer.
    struct ggml_tensor * hc_head_norm      = nullptr;
    struct ggml_tensor * hc_head_down      = nullptr;
    struct ggml_tensor * hc_head_up        = nullptr;
};

// adopted from upstream in this merge: net-new tensor-holding structs (fork had no equivalent at
// all) for architectures needing a WavTokenizer/Encodec-style posnet/convnext decoder stage or a
// short-convolution block or switchable per-layer LoRA. Confirmed needed: src/llama-load-tensors.cpp
// (the live tensor-creation path, already resolved) references fields from this family.
struct llama_layer_posnet {
    // resnet
    struct ggml_tensor * norm1   = nullptr;
    struct ggml_tensor * norm1_b = nullptr;

    struct ggml_tensor * conv1   = nullptr;
    struct ggml_tensor * conv1_b = nullptr;

    struct ggml_tensor * norm2   = nullptr;
    struct ggml_tensor * norm2_b = nullptr;

    struct ggml_tensor * conv2   = nullptr;
    struct ggml_tensor * conv2_b = nullptr;

    // attention
    struct ggml_tensor * attn_norm   = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;

    struct ggml_tensor * attn_q   = nullptr;
    struct ggml_tensor * attn_q_b = nullptr;

    struct ggml_tensor * attn_k   = nullptr;
    struct ggml_tensor * attn_k_b = nullptr;

    struct ggml_tensor * attn_v   = nullptr;
    struct ggml_tensor * attn_v_b = nullptr;

    struct ggml_tensor * attn_o   = nullptr;
    struct ggml_tensor * attn_o_b = nullptr;

    // normalize
    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;
};

struct llama_layer_convnext {
    struct ggml_tensor * dw   = nullptr;
    struct ggml_tensor * dw_b = nullptr;

    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;

    struct ggml_tensor * pw1   = nullptr;
    struct ggml_tensor * pw1_b = nullptr;

    struct ggml_tensor * pw2   = nullptr;
    struct ggml_tensor * pw2_b = nullptr;

    struct ggml_tensor * gamma = nullptr;
};

struct llama_layer_shortconv {
    struct ggml_tensor * in_proj  = nullptr;
    struct ggml_tensor * conv     = nullptr;
    struct ggml_tensor * out_proj = nullptr;
};

struct llama_layer_switch_lora {
    struct ggml_tensor * a_q    = nullptr;
    struct ggml_tensor * b_q    = nullptr;
    struct ggml_tensor * a_k    = nullptr;
    struct ggml_tensor * b_k    = nullptr;
    struct ggml_tensor * a_v    = nullptr;
    struct ggml_tensor * b_v    = nullptr;
    struct ggml_tensor * a_o    = nullptr;
    struct ggml_tensor * b_o    = nullptr;

    struct ggml_tensor * a_gate = nullptr;
    struct ggml_tensor * b_gate = nullptr;
    struct ggml_tensor * a_up   = nullptr;
    struct ggml_tensor * b_up   = nullptr;
    struct ggml_tensor * a_down = nullptr;
    struct ggml_tensor * b_down = nullptr;
};

// (separating this into llama_layer_enc / llama_layer_dec is a pre-existing wish from the
// original source, not something this merge does.)
// This struct is a UNION of both sides' fields (not a pick-one): src/llama-load-tensors.cpp (the
// live, already-resolved tensor-creation path) references fields from BOTH the fork's own
// tensor-parallel split_* infrastructure AND upstream's newer-architecture tensor fields (DSA,
// Kimi Linear KDA, per-expert NVFP4 scales, etc). Merged by scripted name-diff (not hand-transcribed)
// to avoid a duplicate-member compile error across ~275 fork fields and ~275 upstream fields;
// verified 0 name collisions remain except fields that mean the SAME thing on both sides
// (out_scale, nextn), which are kept once.
struct llama_layer_shared {
    // normalization
    struct ggml_tensor * attn_norm = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;
    struct ggml_tensor * attn_norm_2 = nullptr;
    struct ggml_tensor * attn_norm_2_b = nullptr;
    // BitNet "norm_in" weights (RMS-norm multiplier on a projection's input), optional
    struct ggml_tensor * attn_q_norm_in   = nullptr;
    struct ggml_tensor * attn_k_norm_in   = nullptr;
    struct ggml_tensor * attn_v_norm_in   = nullptr;
    struct ggml_tensor * attn_out_norm_in = nullptr;
    struct ggml_tensor * ffn_gate_norm_in = nullptr;
    struct ggml_tensor * ffn_up_norm_in   = nullptr;
    struct ggml_tensor * ffn_down_norm_in = nullptr;
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_q_norm_b = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;
    struct ggml_tensor * attn_k_norm_b = nullptr;
    struct ggml_tensor * attn_out_norm = nullptr;
    struct ggml_tensor * attn_out_norm_b = nullptr;
    struct ggml_tensor * attn_q_a_norm = nullptr;
    struct ggml_tensor * attn_kv_a_norm = nullptr;
    struct ggml_tensor * attn_sub_norm = nullptr;
    struct ggml_tensor * attn_post_norm = nullptr;
    struct ggml_tensor * ffn_sub_norm = nullptr;
    struct ggml_tensor * attn_norm_cross = nullptr;
    struct ggml_tensor * attn_norm_enc = nullptr;
    struct ggml_tensor * wqkv_gate = nullptr;
    struct ggml_tensor * ffn_post_norm_1 = nullptr; // gemma4
    struct ggml_tensor * ffn_post_norm_2 = nullptr; // gemma4
    struct ggml_tensor * ffn_pre_norm_2  = nullptr; // gemma4
    struct ggml_tensor * ffn_gate_inp_s  = nullptr; // gemma4


    // attention
    struct ggml_tensor * wq = nullptr;
    struct ggml_tensor * wk = nullptr;
    struct ggml_tensor * wv = nullptr;
    struct ggml_tensor * wo = nullptr;
    struct ggml_tensor * wqkv = nullptr;
    struct ggml_tensor * wqk  = nullptr;
    struct ggml_tensor * wkv  = nullptr;
    struct ggml_tensor * wq_a = nullptr;
    struct ggml_tensor * wq_b = nullptr;
    struct ggml_tensor * wkv_a_mqa = nullptr;
    struct ggml_tensor * wkq_a_mqa = nullptr;
    struct ggml_tensor * wkv_b = nullptr;
    struct ggml_tensor * wk_b = nullptr;
    // wk_b in pp_opt-favoring layout [kv_lora_rank, qk_nope, n_head], serialized
    // as "attn_kv_b.weight". Materialized under -sm graph + mla>1; mla=1 skips.
    struct ggml_tensor * wk_b_pp = nullptr;
    struct ggml_tensor * wv_b = nullptr;
    struct ggml_tensor * wkv_latent = nullptr;
    struct ggml_tensor * wo_a = nullptr;
    struct ggml_tensor * wo_b = nullptr;
    struct ggml_tensor * wq_cross = nullptr;
    struct ggml_tensor * wk_cross = nullptr;
    struct ggml_tensor * wv_cross = nullptr;
    struct ggml_tensor * wo_cross = nullptr;
    struct ggml_tensor * wq_enc = nullptr;
    struct ggml_tensor * wk_enc = nullptr;
    struct ggml_tensor * wv_enc = nullptr;
    struct ggml_tensor * wo_enc = nullptr;
    struct ggml_tensor * attn_sinks = nullptr;

    // DFlash2 dynamic grouped-convolution parameters.
    struct ggml_tensor * dflash_attn_conv_base = nullptr;
    struct ggml_tensor * dflash_attn_conv_proj = nullptr;
    struct ggml_tensor * dflash_ffn_conv_base = nullptr;
    struct ggml_tensor * dflash_ffn_conv_proj = nullptr;

    // attention bias
    struct ggml_tensor * bq = nullptr;
    struct ggml_tensor * bk = nullptr;
    struct ggml_tensor * bv = nullptr;
    struct ggml_tensor * bo = nullptr;
    struct ggml_tensor * bqkv = nullptr;

    // K smoothing (llama_model_params::k_smooth_alpha). F32, created at load, NOT in the GGUF:
    //   k_smooth_q [n_embd_head_k, n_head]    multiplies Qcur before RoPE (s)
    //   k_smooth_k [n_embd_head_k, n_head_kv] multiplies Kcur before RoPE (1/s)
    // nullptr = smoothing off for this layer; builders must treat that as a no-op.
    struct ggml_tensor * k_smooth_q = nullptr;
    struct ggml_tensor * k_smooth_k = nullptr;
    struct ggml_tensor * bqk  = nullptr;
    struct ggml_tensor * bkv  = nullptr;

    llama_split_tensor split_attn_norm;
    llama_split_tensor split_attn_post_norm;
    llama_split_tensor split_attn_sinks;
    llama_split_tensor split_wq;
    llama_split_tensor split_wk;
    llama_split_tensor split_wv;
    llama_split_tensor split_wo;
    llama_split_tensor split_wqkv;
    llama_split_tensor split_wqk;
    llama_split_tensor split_wkv;
    llama_split_tensor split_bq;
    llama_split_tensor split_bk;
    llama_split_tensor split_bv;
    llama_split_tensor split_bo;
    llama_split_tensor split_bqkv;
    llama_split_tensor split_bqk;
    llama_split_tensor split_bkv;
    llama_split_tensor split_q_norm;
    llama_split_tensor split_k_norm;
    llama_split_tensor split_sinks;
    llama_split_tensor split_wqkv_gate;

    // MLA per-device shards (-sm graph for DEEPSEEK2/DEEPSEEK4/GLM_DSA/MISTRAL4).
    llama_split_tensor split_wq_a;
    llama_split_tensor split_wq_b;
    llama_split_tensor split_wkv_a_mqa;
    llama_split_tensor split_wk_b;
    llama_split_tensor split_wk_b_pp;
    llama_split_tensor split_wv_b;
    llama_split_tensor split_attn_q_a_norm;
    llama_split_tensor split_attn_kv_a_norm;

    llama_split_tensor split_ssm_wqkv;
    llama_split_tensor split_ssm_wqkv_gate;
    llama_split_tensor split_ssm_in;
    llama_split_tensor split_ssm_conv1d;
    llama_split_tensor split_ssm_conv1d_q;
    llama_split_tensor split_ssm_conv1d_k;
    llama_split_tensor split_ssm_conv1d_v;
    llama_split_tensor split_ssm_dt;
    llama_split_tensor split_ssm_a;
    llama_split_tensor split_ssm_beta_alpha;
    llama_split_tensor split_ssm_beta;
    llama_split_tensor split_ssm_alpha;
    llama_split_tensor split_ssm_f_a;
    llama_split_tensor split_ssm_g_a;
    llama_split_tensor split_ssm_norm;
    llama_split_tensor split_ssm_out;
    struct ggml_tensor * wqkv_gate_b = nullptr; // beam

    // relative position bias
    struct ggml_tensor * attn_rel_b = nullptr;
    struct ggml_tensor * attn_rel_b_enc = nullptr;
    struct ggml_tensor * attn_rel_b_cross = nullptr;

    // normalization
    struct ggml_tensor * ffn_norm = nullptr;
    struct ggml_tensor * ffn_norm_b = nullptr;
    struct ggml_tensor * ffn_post_norm = nullptr;
    struct ggml_tensor * layer_out_norm = nullptr;
    struct ggml_tensor * ffn_norm_ada_down = nullptr; // voxtral realtime
    struct ggml_tensor * ffn_norm_ada_up   = nullptr; // voxtral realtime
    struct ggml_tensor * layer_out_norm_b = nullptr;
    struct ggml_tensor * ffn_norm_exps = nullptr;
    struct ggml_tensor * ffn_norm_enc = nullptr;

    // ff
    struct ggml_tensor * ffn_gate = nullptr; // w1
    struct ggml_tensor * ffn_down = nullptr; // w2
    struct ggml_tensor * ffn_up = nullptr;   // w3
    struct ggml_tensor * ffn_gate_enc = nullptr;
    struct ggml_tensor * ffn_down_enc = nullptr;
    struct ggml_tensor * ffn_up_enc = nullptr;

    llama_split_tensor split_ffn_up;
    llama_split_tensor split_ffn_gate;
    llama_split_tensor split_ffn_down;
    llama_split_tensor split_ffn_norm;
    llama_split_tensor split_ffn_up_gate;
    llama_split_tensor split_ffn_post_norm;
    llama_split_tensor split_ffn_post_norm_1;
    llama_split_tensor split_ffn_post_norm_2;
    llama_split_tensor split_ffn_pre_norm_2;

    // ff MoE
    struct ggml_tensor * ffn_gate_inp = nullptr;
    struct ggml_tensor * ffn_gate_exps = nullptr;
    struct ggml_tensor * ffn_down_exps = nullptr;
    struct ggml_tensor * ffn_up_exps  = nullptr;
    struct ggml_tensor * ffn_up_gate_exps  = nullptr;

    llama_split_tensor split_ffn_gate_inp;
    llama_split_tensor split_ffn_up_exps;
    llama_split_tensor split_ffn_gate_exps;
    llama_split_tensor split_ffn_down_exps;
    llama_split_tensor split_ffn_up_gate_exps;

    // ff MoE bias
    struct ggml_tensor * ffn_gate_inp_b = nullptr;
    struct ggml_tensor * ffn_gate_exps_b = nullptr;
    struct ggml_tensor * ffn_down_exps_b = nullptr;
    struct ggml_tensor * ffn_up_exps_b = nullptr;
    struct ggml_tensor * ffn_up_gate_exps_b = nullptr;
    struct ggml_tensor * ffn_gate_exps_b_dup = nullptr;
    struct ggml_tensor * ffn_down_exps_b_dup = nullptr;
    struct ggml_tensor * ffn_up_exps_b_dup = nullptr;
    struct ggml_tensor * ffn_down_exps_s = nullptr;

    // ff shared expert (shexp)
    struct ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct ggml_tensor * ffn_gate_shexp = nullptr;
    struct ggml_tensor * ffn_down_shexp = nullptr;
    struct ggml_tensor * ffn_up_shexp = nullptr;

    // K2 Horizon MoVA
    struct ggml_tensor * attn_v_gate   = nullptr;
    struct ggml_tensor * attn_v_gate_b = nullptr;
    struct ggml_tensor * attn_v_exps   = nullptr;

    llama_split_tensor split_ffn_up_shexp;
    llama_split_tensor split_ffn_gate_shexp;
    llama_split_tensor split_ffn_down_shexp;
    llama_split_tensor split_ffn_gate_inp_shexp;

    llama_split_tensor split_ffn_gate_inp_b;
    llama_split_tensor split_ffn_gate_exps_b;
    llama_split_tensor split_ffn_down_exps_b;
    llama_split_tensor split_ffn_up_exps_b;
    llama_split_tensor split_ffn_up_gate_exps_b;
    llama_split_tensor split_ffn_down_exps_s;
    llama_split_tensor split_ffn_gate_inp_s;

    // ff bias
    // RESTORED 2026-09-08: these three were lost in the merge's hand-resolution of this struct.
    // BOTH merge parents carry them byte-identically -- 279b4bfaf^1:src/llama-model.h:341-343 and
    // 279b4bfaf^2:src/llama-model.h:359-361 -- so no side had anything the other lacked and no
    // merge decision was ever involved here. Their SPLIT companions (split_ffn_gate_b,
    // split_ffn_down_b, split_ffn_up_b) survived intact a few lines below, which is what makes the
    // deletion look deliberate at a glance and is exactly why it was worth checking both parents.
    // Every consumer already referenced these names as if they existed; restoring the declaration
    // is the whole fix. Cascade cleared: 14 files, ~70 diagnostics, including the misleading
    // "llm_build_ffn does not take N arguments" errors -- those fire because one ARGUMENT's type
    // fails to resolve, not because the arity is wrong.
    struct ggml_tensor * ffn_gate_b = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr; // b2
    struct ggml_tensor * ffn_up_b   = nullptr; // b3

    struct ggml_tensor * ffn_act = nullptr;
    struct ggml_tensor * ffn_exp_probs_b = nullptr;
    struct ggml_tensor * ffn_exp_probs_b_vl = nullptr;
    struct ggml_tensor * ffn_gate_tid2eid = nullptr;

    llama_split_tensor split_ffn_gate_b;
    llama_split_tensor split_ffn_down_b;
    llama_split_tensor split_ffn_up_b;
    llama_split_tensor split_ffn_act;
    llama_split_tensor split_ffn_exp_probs_b;

    // misc
    struct ggml_tensor * per_layer_inp_gate;
    struct ggml_tensor * per_layer_proj;
    struct ggml_tensor * per_layer_post_norm;

    // mamba proj
    struct ggml_tensor * ssm_in = nullptr;
    struct ggml_tensor * ssm_x = nullptr;
    struct ggml_tensor * ssm_dt = nullptr;
    struct ggml_tensor * ssm_out = nullptr;
    struct ggml_tensor * ssm_norm = nullptr;
    struct ggml_tensor * ssm_beta_alpha = nullptr;
    struct ggml_tensor * ssm_alpha = nullptr;
    struct ggml_tensor * ssm_beta = nullptr;
    struct ggml_tensor * ssm_f_a = nullptr;
    struct ggml_tensor * ssm_g_a = nullptr;
    // S69 kimi-k3
    struct ggml_tensor * ssm_f_b = nullptr;         // KDA gate, second factor
    // S69 LFM2 short-convolution layer. d_conv = n_shortconv_l_cache - 1.
    struct ggml_tensor * shortconv_conv = nullptr;      // [l_cache, n_embd] conv kernel
    struct ggml_tensor * shortconv_in_proj = nullptr;   // [n_embd, 3*n_embd] -> b, c, x
    struct ggml_tensor * shortconv_out_proj = nullptr;  // [n_embd, n_embd]
    // inkling (S69 P4). SOURCE-READ: src__llama-model.h:595-602 of the saved upstream branch (MIT).
    struct ggml_tensor * wr             = nullptr; // attn_r  [n_embd, n_head*d_rel]
    struct ggml_tensor * attn_rel_proj  = nullptr; // [rel_extent, d_rel] (checkpoint [d_rel, E] orientation), F32
    struct ggml_tensor * shortconv_k    = nullptr; // [K, n_head_kv*head_dim], F32
    struct ggml_tensor * shortconv_v    = nullptr; // [K, n_head_kv*head_dim], F32
    struct ggml_tensor * shortconv_attn = nullptr; // [K, n_embd], F32
    struct ggml_tensor * shortconv_mlp  = nullptr; // [K, n_embd], F32
    struct ggml_tensor * ffn_gscale     = nullptr; // F32 [1]
    // In-Place TTT (row S69-FORK-INPLACE-TTT-2026-10-26-01)
    struct ggml_tensor * ffn_ttt_conv   = nullptr; // depthwise conv taps, ne {n_embd, 5}, F32
    struct ggml_tensor * ffn_ttt_proj   = nullptr; // target projection P stored TRANSPOSED, ne {n_embd, n_embd}; optional
    struct ggml_tensor * attn_res_score = nullptr;  // cross-layer residual, pre-attention
    struct ggml_tensor * ffn_res_score = nullptr;   // cross-layer residual, pre-FFN
    struct ggml_tensor * ffn_routed_down = nullptr; // latent MoE: hidden -> latent
    struct ggml_tensor * ffn_routed_up = nullptr;   // latent MoE: latent -> hidden
    struct ggml_tensor * ffn_routed_norm = nullptr; // latent MoE: norm on expert output
    // GLM-5.3-Flash (kimi-k3) low-rank KDA parameterization: f = f_b(f_a(x)), g = g_b(g_a(x))
    struct ggml_tensor * ssm_g_b = nullptr;

    // mamba
    struct ggml_tensor * ssm_conv1d = nullptr;
    struct ggml_tensor * ssm_conv1d_q = nullptr;
    struct ggml_tensor * ssm_conv1d_k = nullptr;
    struct ggml_tensor * ssm_conv1d_v = nullptr;
    struct ggml_tensor * ssm_a = nullptr;
    struct ggml_tensor * ssm_d = nullptr;

    // mamba bias
    struct ggml_tensor * ssm_conv1d_b = nullptr;
    struct ggml_tensor * ssm_dt_b = nullptr;

    // DSA (deepseek sparse attention)
    struct ggml_tensor * indexer_k_norm   = nullptr;
    struct ggml_tensor * indexer_k_norm_b = nullptr;
    struct ggml_tensor * indexer_proj     = nullptr;
    struct ggml_tensor * indexer_attn_k   = nullptr;
    struct ggml_tensor * indexer_attn_q_b = nullptr; // note: for lora a/b, not bias
    struct ggml_tensor * indexer_comp_wkv   = nullptr;
    struct ggml_tensor * indexer_comp_wgate = nullptr;
    struct ggml_tensor * indexer_comp_ape   = nullptr;
    struct ggml_tensor * indexer_comp_norm  = nullptr;
    struct ggml_tensor * attn_kv_norm       = nullptr;
    struct ggml_tensor * hc_attn_base     = nullptr;
    struct ggml_tensor * hc_attn_fn       = nullptr;
    struct ggml_tensor * hc_attn_scale    = nullptr;
    struct ggml_tensor * hc_ffn_base      = nullptr;
    struct ggml_tensor * hc_ffn_fn        = nullptr;
    struct ggml_tensor * hc_ffn_scale     = nullptr;

    // qwen4exp low-rank hyper-connections
    struct ggml_tensor * hc_attn_norm     = nullptr;
    struct ggml_tensor * hc_attn_down     = nullptr;
    struct ggml_tensor * hc_attn_up       = nullptr;
    struct ggml_tensor * hc_attn_inject   = nullptr;
    struct ggml_tensor * hc_ffn_norm      = nullptr;
    struct ggml_tensor * hc_ffn_down      = nullptr;
    struct ggml_tensor * hc_ffn_up        = nullptr;
    struct ggml_tensor * hc_ffn_inject    = nullptr;

    // qwen4exp QSA indexer
    struct ggml_tensor * indexer_q_proj   = nullptr;
    struct ggml_tensor * indexer_k_proj   = nullptr;
    struct ggml_tensor * indexer_q_norm   = nullptr;

    // qwen4exp per-layer n-gram embedding (PLE); present on ple layers only
    struct ggml_tensor * ple_key          = nullptr;
    struct ggml_tensor * ple_value        = nullptr;
    struct ggml_tensor * ple_norm_key     = nullptr;
    struct ggml_tensor * ple_norm_query   = nullptr;
    struct ggml_tensor * ple_norm_conv    = nullptr;
    struct ggml_tensor * ple_conv1d       = nullptr;
    struct ggml_tensor * attn_comp_wkv     = nullptr;
    struct ggml_tensor * attn_comp_wgate   = nullptr;
    struct ggml_tensor * attn_comp_ape     = nullptr;
    struct ggml_tensor * attn_comp_norm    = nullptr;

    // n-gram keyed table, a few rows read per token. Only ever touched through get_rows,
    // so it must never be copied, repacked or offloaded.
    struct ggml_tensor * engram_embd = nullptr;
    struct ggml_tensor * engram_k    = nullptr;
    struct ggml_tensor * engram_q    = nullptr;
    struct ggml_tensor * engram_wkv  = nullptr;

    // long rope factors
    struct ggml_tensor * rope_long  = nullptr;
    struct ggml_tensor * rope_short = nullptr;
    struct ggml_tensor * rope_freqs = nullptr;

    llama_split_tensor split_rope_freqs;

    // bitnet scale
    struct ggml_tensor * wq_scale = nullptr;
    struct ggml_tensor * wk_scale = nullptr;
    struct ggml_tensor * wv_scale = nullptr;
    struct ggml_tensor * wo_scale = nullptr;
    struct ggml_tensor * ffn_gate_scale = nullptr;
    struct ggml_tensor * ffn_up_scale = nullptr;
    struct ggml_tensor * ffn_down_scale = nullptr;
    struct ggml_tensor * out_scale = nullptr; // gemma4 layer output scale
    // DiffusionGemma encoder-mode per-layer output scale (applied to PROMPT positions only; the
    // decoder scalar above is applied to CANVAS positions). Both come from the same shared stack.
    struct ggml_tensor * enc_out_scale = nullptr;

    llama_split_tensor split_out_scale;

    struct llama_layer_nextn nextn;

    // openPangu-2.0: MoME causal convs + learned static param sink + mHC + block post-norm
    struct ggml_tensor * qa_conv          = nullptr;
    struct ggml_tensor * kv_conv          = nullptr; // compresskv_conv
    struct ggml_tensor * o_conv           = nullptr;
    struct ggml_tensor * param_sink_kv    = nullptr;
    struct ggml_tensor * param_sink_k_pe  = nullptr;
    struct ggml_tensor * param_sink_blk   = nullptr;
    struct ggml_tensor * param_sink_lat_t = nullptr;
    struct ggml_tensor * block_post_norm  = nullptr;
    struct ggml_tensor * mhc_attn_phi     = nullptr;
    struct ggml_tensor * mhc_attn_alpha   = nullptr;
    struct ggml_tensor * mhc_attn_beta    = nullptr;
    struct ggml_tensor * mhc_attn_gamma   = nullptr;
    struct ggml_tensor * mhc_mlp_phi      = nullptr;
    struct ggml_tensor * mhc_mlp_alpha    = nullptr;
    struct ggml_tensor * mhc_mlp_beta     = nullptr;
    struct ggml_tensor * mhc_mlp_gamma    = nullptr;


    // A STRAY `};` STOOD HERE and was removed 2026-09-08. The merge appended the
    // "adopted from upstream" field block BELOW struct llama_layer's closing brace instead of
    // inside it, and left the original closer in place further down (now the struct's only one).
    // The struct therefore ENDED at this line while ~155 more lines of member declarations --
    // the upstream-adopted fields immediately below, plus the fork's own ssm_*/index_*/posnet/
    // convnext/shortconv/switch_lora members -- sat at FILE scope.
    //
    // Why it hid: the file's braces still BALANCED (39 open, 39 close). It was only found by
    // walking depth and asking where each named scope RETURNS to its opening depth -- llama_layer
    // opened at 289 and closed at 643 while its members ran to 798. A brace count cannot see this;
    // it needs the structure.
    //
    // It is also what made the earlier duplicated `enum e_model {` opener look harmless: that
    // duplicate consumed the extra closer, so removing the duplicate is what turned a balanced-
    // but-wrong file into an unbalanced one and finally surfaced this.

    // ---- adopted from upstream in this merge: fields with no fork equivalent ----
    struct ggml_tensor * ssm_dt_norm     = nullptr;
    struct ggml_tensor * ssm_b_norm      = nullptr;
    struct ggml_tensor * ssm_c_norm      = nullptr;
    struct ggml_tensor * wg        = nullptr;
    struct ggml_tensor * wqkv_b    = nullptr;
    struct ggml_tensor * ffn_gate_up_exps  = nullptr;
    struct ggml_tensor * ffn_gate_up_exps_b = nullptr;

    // ff MoE per-expert scales (NVFP4 per-tensor scale2)
    struct ggml_tensor * ffn_gate_exps_s   = nullptr;
    struct ggml_tensor * ffn_up_exps_s     = nullptr;

    // ff MoE latent proj
    struct ggml_tensor * ffn_latent_down = nullptr;
    struct ggml_tensor * ffn_latent_up   = nullptr;

    // ff MoE norms around the experts, in latent space (muse-spark)
    struct ggml_tensor * ffn_pre_exp_norm  = nullptr;
    struct ggml_tensor * ffn_post_exp_norm = nullptr;

    // norm preserving residual coefficients (muse-spark)
    struct ggml_tensor * attn_resid_alpha = nullptr;
    struct ggml_tensor * attn_resid_beta  = nullptr;
    struct ggml_tensor * ffn_resid_alpha  = nullptr;
    struct ggml_tensor * ffn_resid_beta   = nullptr;

    // ff adjugate experts (chexps)
    struct ggml_tensor * ffn_gate_chexps     = nullptr;
    struct ggml_tensor * ffn_down_chexps     = nullptr;
    struct ggml_tensor * ffn_up_chexps       = nullptr;

    // LiLiCorr draft head (dflash.cpp)
    struct ggml_tensor * lilicorr_attn_norm       = nullptr;
    struct ggml_tensor * lilicorr_mlp_norm        = nullptr;
    struct ggml_tensor * lilicorr_attn_in_proj    = nullptr;
    struct ggml_tensor * lilicorr_attn_in_proj_b  = nullptr;
    struct ggml_tensor * lilicorr_attn_out_proj   = nullptr;
    struct ggml_tensor * lilicorr_attn_out_proj_b = nullptr;
    struct ggml_tensor * lilicorr_attn_out_proj_s = nullptr;
    struct ggml_tensor * lilicorr_mlp_0           = nullptr;
    struct ggml_tensor * lilicorr_mlp_0_b         = nullptr;
    struct ggml_tensor * lilicorr_mlp_0_s         = nullptr;
    struct ggml_tensor * lilicorr_mlp_2           = nullptr;
    struct ggml_tensor * lilicorr_mlp_2_b         = nullptr;
    struct ggml_tensor * lilicorr_mlp_2_s         = nullptr;

    // rwkv
    struct ggml_tensor * time_mix_w1         = nullptr;
    struct ggml_tensor * time_mix_w2         = nullptr;
    struct ggml_tensor * time_mix_lerp_x     = nullptr;
    struct ggml_tensor * time_mix_lerp_w     = nullptr;
    struct ggml_tensor * time_mix_lerp_k     = nullptr;
    struct ggml_tensor * time_mix_lerp_v     = nullptr;
    struct ggml_tensor * time_mix_lerp_r     = nullptr;
    struct ggml_tensor * time_mix_lerp_g     = nullptr;
    struct ggml_tensor * time_mix_lerp_fused = nullptr;

    struct ggml_tensor * time_mix_first        = nullptr;
    struct ggml_tensor * time_mix_decay        = nullptr;
    struct ggml_tensor * time_mix_decay_w1     = nullptr;
    struct ggml_tensor * time_mix_decay_w2     = nullptr;
    struct ggml_tensor * time_mix_key          = nullptr;
    struct ggml_tensor * time_mix_key_b        = nullptr;
    struct ggml_tensor * time_mix_value        = nullptr;
    struct ggml_tensor * time_mix_value_b      = nullptr;
    struct ggml_tensor * time_mix_receptance   = nullptr;
    struct ggml_tensor * time_mix_receptance_b = nullptr;
    struct ggml_tensor * time_mix_gate         = nullptr;

    // mmfreelm (MatMul-free LM)
    struct ggml_tensor * hgrn_i = nullptr;
    struct ggml_tensor * hgrn_f = nullptr;
    struct ggml_tensor * hgrn_g = nullptr;
    struct ggml_tensor * hgrn_i_norm = nullptr;
    struct ggml_tensor * hgrn_f_norm = nullptr;
    struct ggml_tensor * hgrn_g_norm = nullptr;
    struct ggml_tensor * hgrn_o_norm = nullptr;
    struct ggml_tensor * hgrn_gnorm = nullptr;
    struct ggml_tensor * hgrn_lower_bound = nullptr;
    struct ggml_tensor * ffn_in_norm = nullptr;
    struct ggml_tensor * ffn_down_in_norm = nullptr;

    // rwkv7
    struct ggml_tensor * time_mix_w0         = nullptr;
    struct ggml_tensor * time_mix_a0         = nullptr;
    struct ggml_tensor * time_mix_a1         = nullptr;
    struct ggml_tensor * time_mix_a2         = nullptr;
    struct ggml_tensor * time_mix_v0         = nullptr;
    struct ggml_tensor * time_mix_v1         = nullptr;
    struct ggml_tensor * time_mix_v2         = nullptr;
    struct ggml_tensor * time_mix_g1         = nullptr;
    struct ggml_tensor * time_mix_g2         = nullptr;
    struct ggml_tensor * time_mix_k_k        = nullptr;
    struct ggml_tensor * time_mix_k_a        = nullptr;
    struct ggml_tensor * time_mix_r_k        = nullptr;

    struct ggml_tensor * time_mix_ln     = nullptr;
    struct ggml_tensor * time_mix_ln_b   = nullptr;
    struct ggml_tensor * time_mix_output = nullptr;

    struct ggml_tensor * channel_mix_lerp_k = nullptr;
    struct ggml_tensor * channel_mix_lerp_r = nullptr;

    struct ggml_tensor * channel_mix_key        = nullptr;
    struct ggml_tensor * channel_mix_receptance = nullptr;
    struct ggml_tensor * channel_mix_value      = nullptr;

    // bitnet scale
    struct ggml_tensor * wq_s       = nullptr;
    struct ggml_tensor * wk_s       = nullptr;
    struct ggml_tensor * wv_s       = nullptr;
    struct ggml_tensor * wo_s       = nullptr;
    struct ggml_tensor * wqkv_s     = nullptr;
    struct ggml_tensor * wqkv_gate_s = nullptr;
    struct ggml_tensor * ffn_gate_s = nullptr;
    struct ggml_tensor * ffn_up_s   = nullptr;
    struct ggml_tensor * ffn_down_s = nullptr;
    struct ggml_tensor * ffn_gate_shexp_s = nullptr;
    struct ggml_tensor * ffn_up_shexp_s   = nullptr;
    struct ggml_tensor * ffn_down_shexp_s = nullptr;
    struct ggml_tensor * ssm_in_s    = nullptr;
    struct ggml_tensor * ssm_out_s   = nullptr;
    struct ggml_tensor * ssm_alpha_s = nullptr;
    struct ggml_tensor * ssm_beta_s  = nullptr;

    // input scales
    struct ggml_tensor * wq_in_s            = nullptr;
    struct ggml_tensor * wk_in_s            = nullptr;
    struct ggml_tensor * wv_in_s            = nullptr;
    struct ggml_tensor * wo_in_s            = nullptr;
    struct ggml_tensor * wqkv_in_s          = nullptr;
    struct ggml_tensor * wqkv_gate_in_s     = nullptr;
    struct ggml_tensor * ffn_gate_in_s      = nullptr;
    struct ggml_tensor * ffn_up_in_s        = nullptr;
    struct ggml_tensor * ffn_down_in_s      = nullptr;
    struct ggml_tensor * ffn_gate_exps_in_s = nullptr;
    struct ggml_tensor * ffn_down_exps_in_s = nullptr;
    struct ggml_tensor * ffn_up_exps_in_s   = nullptr;
    struct ggml_tensor * ffn_gate_shexp_in_s= nullptr;
    struct ggml_tensor * ffn_up_shexp_in_s  = nullptr;
    struct ggml_tensor * ffn_down_shexp_in_s= nullptr;
    struct ggml_tensor * ssm_in_in_s        = nullptr;
    struct ggml_tensor * ssm_out_in_s       = nullptr;
    struct ggml_tensor * ssm_alpha_in_s     = nullptr;
    struct ggml_tensor * ssm_beta_in_s      = nullptr;
    struct ggml_tensor * altup_correct_coef   = nullptr;
    struct ggml_tensor * altup_correct_scale  = nullptr;
    struct ggml_tensor * altup_predict_coef   = nullptr;
    struct ggml_tensor * altup_router         = nullptr;
    struct ggml_tensor * altup_router_norm    = nullptr;
    struct ggml_tensor * laurel_l             = nullptr;
    struct ggml_tensor * laurel_r             = nullptr;
    struct ggml_tensor * laurel_post_norm     = nullptr;

    // cogvlm
    struct ggml_tensor * visexp_attn_wqkv = nullptr;
    struct ggml_tensor * visexp_attn_wo   = nullptr;
    struct ggml_tensor * visexp_ffn_gate  = nullptr;
    struct ggml_tensor * visexp_ffn_down  = nullptr;
    struct ggml_tensor * visexp_ffn_up    = nullptr;

    // xIELU activation parameters for Apertus
    struct ggml_tensor * ffn_act_alpha_n = nullptr;
    struct ggml_tensor * ffn_act_alpha_p = nullptr;
    struct ggml_tensor * ffn_act_beta    = nullptr;
    struct ggml_tensor * ffn_act_eps     = nullptr;

    // Kimi Linear KDA (using ssm_ prefix for consistency)
    // Note: ssm_dt_b already exists above (mamba bias), reused for Kimi dt_bias
    struct ggml_tensor * ssm_q_conv = nullptr;
    struct ggml_tensor * ssm_k_conv = nullptr;
    struct ggml_tensor * ssm_v_conv = nullptr;
    struct ggml_tensor * ssm_o_norm = nullptr;

    // kimi-k3
    struct ggml_tensor * ssm_g           = nullptr; // full-rank KDA gate (replaces ssm_g_a/ssm_g_b)

    // glm5-next k-pool indexer
    struct ggml_tensor * indexer_kpool_gate = nullptr;
    struct ggml_tensor * indexer_kpool_ape  = nullptr;

    // deepseek41
    struct ggml_tensor * engram_kv   = nullptr;
    struct ggml_tensor * engram_gate = nullptr; // q_weight*k_weight folded

    // MSA
    struct ggml_tensor * index_q_proj = nullptr;
    struct ggml_tensor * index_k_proj = nullptr;
    struct ggml_tensor * index_q_norm = nullptr;
    struct ggml_tensor * index_k_norm = nullptr;

    // qwen4exp fused weights
    struct ggml_tensor * hc_attn_down_inject = nullptr; // [hc_attn_down | hc_attn_inject] in one weight
    struct ggml_tensor * hc_ffn_down_inject  = nullptr; // [hc_ffn_down | hc_ffn_inject] in one weight
    struct ggml_tensor * wqkv_z              = nullptr; // [wqkv | wqkv_gate] in one weight
    struct ggml_tensor * ssm_beta_alpha_cat  = nullptr; // [ssm_beta | ssm_alpha] in one weight, unlike the grouped ssm_beta_alpha

    struct llama_layer_posnet posnet;

    struct llama_layer_convnext convnext;

    struct llama_layer_shortconv shortconv;

    struct llama_layer_switch_lora switch_lora;
};

// SPLIT 2026-09-09. Everything above is the layer's LOADED WEIGHTS: raw ggml_tensor pointers into
// buffers the model owns. Copying that is SHARING, which is what upstream's all-raw-pointer
// llama_layer did implicitly and what src/models/nanbeige.cpp:70 depends on --
//     layers[i + j * n_phys] = layers[i];   // "Share physical weights across loops; each slot
//                                           //  still has its own KV index."
// nanbeige loads n_phys PHYSICAL layers and replicates them across n_loops LOGICAL slots.
//
// Below are this fork's NINE OWNING members: lazily-computed derived tensors (MLA wk_b/wv_b, the
// parameter sinks, and the per-device replicas for the -sm graph). std::unique_ptr is not
// copy-assignable, so their presence in the struct made the compiler delete llama_layer's copy
// assignment outright, and upstream's file could not compile.
//
// The copy operations below therefore SHARE THE WEIGHTS AND DO NOT DUPLICATE OWNERSHIP. That is
// not a convenience: a derived tensor belongs to the layer that computed it, and a copy that
// duplicated the pointer would hand two layers one buffer for one of them to free. At nanbeige's
// call site all nine are empty on both sides anyway -- the copy happens inside load_arch_tensors,
// before anything computes them -- so nothing is lost there either.
//
// WHY A BASE SPLIT AND NOT A HAND-WRITTEN 307-FIELD operator=: a hand-written one is correct today
// and silently incorrect the first time a field is added and not listed. A new weight field lands
// in llama_layer_shared and is copied automatically.
//
// Checked over all 1,533 source files before doing this, because a base class changes layout and
// breaks aggregate initialisation: aggregate `llama_layer{...}` 0 sites, memcpy/memset on a layer
// 0, offsetof(llama_layer) 0, is_trivially_*<llama_layer> 0, sizeof(llama_layer) 0.
struct llama_layer : llama_layer_shared {
    std::unique_ptr<ggml_tensor> computed_wk_b;
    std::unique_ptr<ggml_tensor> computed_wk_b_pp;
    std::unique_ptr<ggml_tensor> computed_wv_b;
    std::unique_ptr<ggml_tensor> computed_wkv_b;
    std::unique_ptr<ggml_tensor> computed_param_sink_blk;
    std::unique_ptr<ggml_tensor> computed_param_sink_lat_t;

    // Per-device replicas of computed wk_b/wv_b (-sm graph). Buffers owned via model.bufs.
    std::vector<std::unique_ptr<ggml_tensor>> computed_wk_b_replicas;
    std::vector<std::unique_ptr<ggml_tensor>> computed_wk_b_pp_replicas;
    std::vector<std::unique_ptr<ggml_tensor>> computed_wv_b_replicas;

    llama_layer() = default;
    ~llama_layer() = default;

    // Moves transfer ownership normally.
    llama_layer(llama_layer &&) noexcept = default;
    llama_layer & operator=(llama_layer &&) noexcept = default;

    // Copies SHARE the weights and leave the owning members of the destination alone. The
    // destination keeps whatever it had computed; it does not inherit, and does not free, the
    // source's.
    llama_layer(const llama_layer & other) : llama_layer_shared(other) {}
    llama_layer & operator=(const llama_layer & other) {
        llama_layer_shared::operator=(other);
        return *this;
    }
};

struct llama_lora_adapter;
struct llama_adapter_lora;   // upstream new-name family (src/llama-adapter.h)

struct rpc_device {
    std::string endpoint;
    uint32_t device;
    // Free memory as the server reported it, fetched once per model load. Every uncached query is a
    // fresh TCP connect + HELLO (the RPC client keeps a socket only while a reference is held), and the
    // loader asks per device, per enumeration and per layer split -- MEASURED 370 connections per load
    // of a 30-layer model, independent of tokens generated.
    mutable bool   free_cached = false;
    mutable size_t free_bytes  = 0;
};

struct llama_cparams;

// Prism ternary activation-side Hadamard transform
struct llama_hadamard_transform {
    struct ggml_tensor * signs = nullptr; // [ne0]
    int64_t block_size = 0;               // Hadamard block size, applied with ggml_hadamard
    int64_t perm_hd  = 1;                 // GDN v-grouped reshape (head dim)
    int64_t perm_nk  = 1;                 // GDN v-grouped reshape (n groups)
    int64_t perm_rep = 1;                 // GDN v-grouped reshape (repeat count)
    bool operator==(const llama_hadamard_transform & other) const {
        return block_size == other.block_size && perm_hd == other.perm_hd && perm_nk == other.perm_nk && perm_rep == other.perm_rep
            && signs == other.signs;
    }
};

// SPLIT_MODE_TENSOR (ggml meta device): userdata of the split-state callback (upstream llama-model.h:658-663)
struct llama_meta_device_get_split_state_userdata {
    size_t                     n_devices = 0;       // 0 until a SPLIT_MODE_TENSOR load fills it (fork: deferred)
    const struct llama_model * model     = nullptr;
};

struct ggml_backend_meta_split_state llama_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata);

struct llama_prec_policy {
    // the key is the weight tensor `res->src[0]`, stores the recommended accumulation type of the op (unused for now)
    // TODO: migrate ad-hoc ggml_prec_set_acc() calls to this container + update apply() to use it
    std::unordered_map<const ggml_tensor *, ggml_prec> prec_acc;

    // the key is the weight tensor `res->src[0]`, stores the recommended activation precision type
    std::unordered_map<const ggml_tensor *, ggml_prec> prec_src1;

    bool apply(ggml_tensor * res) const;

    void load(llama_model_loader & ml, const llama_model & model);
};

// transform of a folded weight, applied to the matmul input: optional sign flip, then the normalized block Hadamard rotation
// Upstream names this llama_hadamard_transform; renamed in this merge (2026-11-04) because the fork's own Prism path
// already defines a different struct of that name (above, used by hadamard_map / the legacy builder). Only the upstream
// class-path (hdmd) uses this one, through llama_hadamard_rotations. Row
// S69-ENGREV-UPSTREAM-MERGE-PRISM-HADAMARD-TWO-IMPLEMENTATIONS-2026-11-04-01 unifies the two.
struct llama_hdmd_transform {
    ggml_tensor * rot;
    ggml_tensor * signs; // nullptr for identity sign mode

    // if perm_rep > 1, permute the input from tiled head order [hd, nk, rep] to grouped order [hd, rep, nk] before signs and rotation
    int64_t perm_hd  = 0;
    int64_t perm_nk  = 0;
    int64_t perm_rep = 0;

    // ssm_out of a gated delta net gets its value heads in tiled order, but the fold used grouped order
    // record the head geometry for that permutation, return false if it does not match the input width
    bool set_gdn_v_perm(int64_t n_in, int64_t n_v, int64_t n_k) {
        if (n_k <= 0 || n_v <= 0 || n_v % n_k != 0 || n_in % n_v != 0) {
            return false;
        }
        perm_hd  = n_in / n_v;
        perm_nk  = n_k;
        perm_rep = n_v / n_k;
        return true;
    }
};
using llama_hadamard_rotations = std::unordered_map<const ggml_tensor *, llama_hdmd_transform>;

struct llama_hadamard {
    // names and sign data come from the GGUF metadata in load_hparams, the transforms are made in load_tensors
    std::unordered_map<std::string, uint32_t> weight_blocks;
    std::unordered_map<std::string, uint32_t> inverse_blocks;

    std::map<uint32_t, std::vector<int32_t>> sign_data;

    bool gdn_v_grouped = false;
    bool tied_output = false;

    llama_hadamard_rotations rot; // folded weight -> activation transform
    llama_hadamard_rotations inv; // latent lookup table -> inverse transform
};

struct llama_model;
// Make hdmd.rot / hdmd.inv (upstream's layout) from the names, sign data and flags already recorded in model.hdmd and the
// model's loaded tensors: one block x block normalized Sylvester-Walsh-Hadamard matrix per (block size, buffer type), one sign
// tensor per (width, buffer type). Returns the number of transforms made; throws std::runtime_error on inconsistent input.
size_t llama_hdmd_fill(llama_model & model);

struct llama_model {
    e_model     type  = MODEL_UNKNOWN;
    llm_arch    arch  = LLM_ARCH_UNKNOWN;
    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    // Prism ternary Hadamard rotations, keyed by GGUF weight name (empty for other models)
    std::unordered_map<std::string, llama_hadamard_transform> hadamard_map;
    const llama_hadamard_transform * hadamard_rotation(const struct ggml_tensor * t) const;

    // Upstream Prism path (class-path graphs, llm_graph_context): filled by upstream's load_hparams_hadamard.
    // Fork Prism path (legacy builder) is hadamard_map above. Two implementations, one per graph path; see row
    // S69-ENGREV-UPSTREAM-MERGE-PRISM-HADAMARD-TWO-IMPLEMENTATIONS-2026-11-04-01.
    // per-tensor activation precision policy
    llama_prec_policy prec_policy;

    llama_hadamard hdmd;

    // see llama_model_set_attn_rot
    bool attn_rot = true;

    // NVFP4 per-tensor scale (S69-A-PICK-NVFP4-TWO-LEVEL-SCALE, 2026-10-05). A GGUF written by llama-quantize
    // (--nvfp4-scale two) or by the ModelOpt converter carries "<name>.scale" (F32, one value) next to an NVFP4
    // "<name>.weight"; value = e2m1 * ue4m3(block) * scale. The loader attaches that tensor to the weight here and the
    // mat-mul funnels of both graph builders (llm_graph_context::build_lora_mm, llm_build_context::llm_build_lora_mm)
    // multiply the product by it. Only plain 2-D weights are attached; empty for every other model.
    // Architectures whose graph was TESTED to multiply the attached scale (single source for the loader and llama-quantize
    // --nvfp4-scale auto). Architectures built by llm_graph_context (src/models/*.cpp) take the scale through their own
    // explicit *_s tensors instead and never consult this map.
    static bool arch_applies_nvfp4_tensor_scale(llm_arch a) {
        switch (a) {
            case LLM_ARCH_LLAMA:
            case LLM_ARCH_QWEN2:
                return true;
            default:
                return false;
        }
    }
    std::unordered_map<const struct ggml_tensor *, struct ggml_tensor *> weight_scale_map;
    struct ggml_tensor * weight_scale(const struct ggml_tensor * w) const {
        if (weight_scale_map.empty() || !w) {
            return nullptr;
        }
        auto it = weight_scale_map.find(w);
        return it == weight_scale_map.end() ? nullptr : it->second;
    }

    struct ggml_tensor * tok_embd = nullptr;
    struct ggml_tensor * type_embd = nullptr;
    struct ggml_tensor * pos_embd = nullptr;
    struct ggml_tensor * tok_norm = nullptr;
    struct ggml_tensor * tok_norm_b = nullptr;

    struct ggml_tensor * tok_embd_per_layer = nullptr;
    struct ggml_tensor * per_layer_model_proj = nullptr;
    struct ggml_tensor * per_layer_proj_norm = nullptr;

    struct ggml_tensor * mtp_pre_proj = nullptr;
    struct ggml_tensor * mtp_post_proj = nullptr;
    struct ggml_tensor * mtp_token_ordering = nullptr;
    struct ggml_tensor * mtp_centroids = nullptr;
    struct ggml_tensor * dflash_fc = nullptr;
    struct ggml_tensor * dflash_hidden_norm = nullptr;
    struct ggml_tensor * dflash_selector_prev = nullptr;
    struct ggml_tensor * dflash_selector_next = nullptr;
    struct ggml_tensor * dflash_selector_hidden = nullptr;
    std::vector<struct ggml_tensor *> dflash_aux_hidden_norms;
    struct ggml_tensor * dspark_markov_w1 = nullptr;
    struct ggml_tensor * dspark_markov_w2 = nullptr;
    struct ggml_tensor * dspark_conf_proj = nullptr;
    struct ggml_tensor * dspark_conf_proj_b = nullptr;

    // ---- DiffusionGemma (LLM_ARCH_DIFFUSION_GEMMA) -------------------------------------------
    // model-level self-conditioning gated MLP (decoder-only, optional: absent == zero-SC forward)
    struct ggml_tensor * sc_pre_norm = nullptr;
    struct ggml_tensor * sc_gate     = nullptr;
    struct ggml_tensor * sc_up       = nullptr;
    struct ggml_tensor * sc_down     = nullptr;

    // Per-request diffusion state, set through the llama_diffusion_* C API before llama_decode.
    // `mutable` because the graph is built from a const llama_model & and lazily allocates the
    // device-side stores below; upstream carried the same state on a polymorphic model subclass,
    // which this fork's plain-struct llama_model cannot express.
    enum dg_pkv_phase_t { DG_PKV_UNIFIED = 0, DG_PKV_PREFILL = 1, DG_PKV_DECODE = 2 };

    // self-conditioning: sc_enabled false == byte-identical to the zero-SC forward; sc_use is a
    // {0,1} gate (a block's FIRST denoising step must pass 0); sc_logits_ptr is a borrowed host
    // [n_vocab * C] buffer that must stay alive through the next llama_decode.
    mutable const float * dg_sc_logits_ptr = nullptr;
    mutable float         dg_sc_use        = 0.0f;
    mutable float         dg_sc_temp_inv   = 1.0f;
    mutable bool          dg_sc_enabled    = false;

    // tok_embd dequantized + transposed to [n_vocab, n_embd] F16 in a device weights buffer, so the
    // per-step SC soft-embedding is one on-device matmul instead of a host gather. Built once.
    mutable struct ggml_tensor  * dg_sc_embT     = nullptr;
    mutable struct ggml_context * dg_sc_embT_ctx = nullptr;
    mutable ggml_backend_buffer_t dg_sc_embT_buf = nullptr;

    // device-resident SC: keep the previous step's raw canvas logits on-device instead of a
    // per-step host round-trip. Same F32 values, so the forward stays bit-for-bit identical.
    mutable bool                  dg_sc_device_resident = false;
    mutable struct ggml_tensor  * dg_sc_dev     = nullptr;   // [n_vocab, dg_sc_dev_C] F32
    mutable struct ggml_context * dg_sc_dev_ctx = nullptr;
    mutable ggml_backend_buffer_t dg_sc_dev_buf = nullptr;
    mutable int64_t               dg_sc_dev_C   = 0;         // allocated canvas capacity (grow-only)

    // prompt-KV caching: the prompt's per-layer K,V are step-invariant, so compute them once per
    // block (PREFILL) and reuse them across denoising steps (DECODE) instead of re-running the whole
    // [prompt|canvas] forward. UNIFIED is the default AND the safety fallback.
    mutable int                   dg_pkv_phase       = DG_PKV_UNIFIED;
    mutable int64_t               dg_pkv_P           = 0;    // prompt length of the current block
    mutable int64_t               dg_pkv_prefill_off = 0;    // PREFILL: global start of this chunk
    mutable int64_t               dg_pkv_cap         = 0;    // allocated capacity (max P)
    mutable std::vector<struct ggml_tensor *> dg_pkv_k;      // per layer [hd, n_head_kv, cap]
    mutable std::vector<struct ggml_tensor *> dg_pkv_v;
    mutable struct ggml_context * dg_pkv_ctx = nullptr;
    mutable ggml_backend_buffer_t dg_pkv_buf = nullptr;

    // free every lazily-allocated DiffusionGemma buffer (no-op on other archs)
    void dg_free_buffers();

    struct ggml_tensor * output_norm = nullptr;
    struct ggml_tensor * output_norm_b = nullptr;
    struct ggml_tensor * output_in_norm = nullptr; // mmfreelm: BitLinear input norm of lm_head
    struct ggml_tensor * output = nullptr;
    struct ggml_tensor * output_b = nullptr;
    struct ggml_tensor * output_res_score = nullptr; // S69 kimi-k3: final residual score
    // S69 #5/#32c — EAGLE3 draft head: fc=encoder feature-fusion, d2t=draft->target vocab map
    struct ggml_tensor * fc  = nullptr;
    struct ggml_tensor * d2t = nullptr;
    struct ggml_tensor * output_norm_enc = nullptr;
    struct ggml_tensor * output_mtp = nullptr;
    struct ggml_tensor * hc_head_base = nullptr;
    struct ggml_tensor * hc_head_fn = nullptr;
    struct ggml_tensor * hc_head_scale = nullptr;

    // engram hash constants, indexed [engram layer][ngram] and [engram layer][(ngram-1)*n_head
    // + head]. They index straight into engram_embd, so the loader validates their counts.
    std::vector<uint64_t> engram_multipliers;
    std::vector<uint64_t> engram_primes;
    std::vector<uint64_t> engram_offsets;
    std::vector<uint32_t> engram_token_map;
    uint32_t              engram_pad_id = 0;

    // qwen4exp: final low-rank hyper-connection mix, plus the n-gram embedding table
    struct ggml_tensor * hc_head_norm = nullptr;
    struct ggml_tensor * hc_head_down = nullptr;
    struct ggml_tensor * hc_head_up   = nullptr;

    // openPangu-2.0: global mHC stream-merge module (non-block)
    struct ggml_tensor * mhc_merge_phi   = nullptr;
    struct ggml_tensor * mhc_merge_alpha = nullptr;
    struct ggml_tensor * mhc_merge_beta  = nullptr;
    struct ggml_tensor * mhc_merge_gamma = nullptr;

    std::unique_ptr<ggml_tensor> output_mtp_ptr;

    // Device-local DFlash IO copies for cross-buffer sharing.
    std::unique_ptr<ggml_tensor> dflash_tok_embd_ptr;
    std::unique_ptr<ggml_tensor> dflash_output_ptr;
    std::unique_ptr<ggml_tensor> dflash_output_mtp_ptr;

    std::unique_ptr<ggml_tensor> qwen4exp_tok_embd_ptr;
    std::unique_ptr<ggml_tensor> qwen4exp_output_ptr;

    llama_split_tensor split_output;
    llama_split_tensor split_output_norm;

    std::vector<llama_layer> layers;

    llama_split_mode split_mode;
    int main_gpu;
    int max_gpu = 0; // max. number of GPUs to use per layer for aplit mode "graph"
    int n_gpu_layers;

    bool mtp; // use mtp if is supported by the Model
    bool swa_compress = false; // value the cache-size fit was computed with
    bool fit = false;

    std::vector<rpc_device> rpc_servers;
    std::vector<int32_t> devices;
    std::vector<int32_t> default_layer_device;
    std::vector<float>   aux_buffer;

    // ---- adopted from upstream in this merge: fields with no fork equivalent ----
    // for classifier models
    std::vector<std::string> classifier_labels;
    std::vector<std::string> decision_labels; // "<arch>.decision.labels" (upstream decision models, /v1/systemone)

    // NVFP4 per-tensor scale2, input_scale for LM head
    struct ggml_tensor * output_s    = nullptr;
    struct ggml_tensor * output_in_s = nullptr;

    // NextN/MTP model-level projections
    struct ggml_tensor * nextn_proj_pre  = nullptr;
    struct ggml_tensor * nextn_proj_post = nullptr;

    // hrm-text initial low-cycle state
    struct ggml_tensor * hrm_z_l_init = nullptr;

    // classifier
    struct ggml_tensor * cls       = nullptr;
    struct ggml_tensor * cls_b     = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;
    struct ggml_tensor * cls_norm  = nullptr;
    struct ggml_tensor * cls_norm_b = nullptr;

    struct ggml_tensor * conv1d   = nullptr;
    struct ggml_tensor * conv1d_b = nullptr;

    // gemma3n altup
    struct ggml_tensor * altup_proj           = nullptr;
    struct ggml_tensor * altup_unembd_proj    = nullptr;
    struct ggml_tensor * per_layer_tok_embd   = nullptr;
    struct ggml_tensor * fc_s = nullptr;
    struct ggml_tensor * fc_in_s = nullptr;
    struct ggml_tensor * dspark_markov_w2_s = nullptr;
    // (dflash_selector_prev / _next / _hidden: declared once, above with the other dflash_* members; the merge had
    //  brought upstream's copy here too)

    struct ggml_tensor * lilicorr_token_proj          = nullptr;
    struct ggml_tensor * lilicorr_token_proj_b        = nullptr;
    struct ggml_tensor * lilicorr_token_proj_s        = nullptr;
    struct ggml_tensor * lilicorr_pass_hidden_proj    = nullptr;
    struct ggml_tensor * lilicorr_pass_hidden_proj_b  = nullptr;
    struct ggml_tensor * lilicorr_pass_hidden_proj_s  = nullptr;
    struct ggml_tensor * lilicorr_feature_norm        = nullptr;
    struct ggml_tensor * lilicorr_feature_norm_b      = nullptr;
    struct ggml_tensor * lilicorr_feature_mlp_1       = nullptr;
    struct ggml_tensor * lilicorr_feature_mlp_1_b     = nullptr;
    struct ggml_tensor * lilicorr_feature_mlp_3       = nullptr;
    struct ggml_tensor * lilicorr_feature_mlp_3_b     = nullptr;
    struct ggml_tensor * lilicorr_slot_embedding      = nullptr;
    struct ggml_tensor * lilicorr_rank_embedding      = nullptr;
    struct ggml_tensor * lilicorr_attention_bias      = nullptr;
    struct ggml_tensor * lilicorr_context_proj        = nullptr;
    struct ggml_tensor * lilicorr_context_proj_b      = nullptr;
    struct ggml_tensor * lilicorr_context_proj_s      = nullptr;
    struct ggml_tensor * lilicorr_output_norm         = nullptr;
    struct ggml_tensor * lilicorr_anchor_norm         = nullptr;
    struct ggml_tensor * lilicorr_factor_input_proj   = nullptr;
    struct ggml_tensor * lilicorr_factor_input_proj_b = nullptr;
    struct ggml_tensor * lilicorr_out_head            = nullptr;
    struct ggml_tensor * lilicorr_out_head_b          = nullptr;
    struct ggml_tensor * lilicorr_in_head             = nullptr;
    struct ggml_tensor * lilicorr_in_head_b           = nullptr;
    struct ggml_tensor * lilicorr_anchor_out_head     = nullptr;
    struct ggml_tensor * lilicorr_anchor_out_head_b   = nullptr;


    // unified vector to store target-model extracted layer ids in eagle3, dflash, etc.
    std::vector<int32_t> target_layer_ids;

    // Dense linear projections for SentenceTransformers models like embeddinggemma. See
    // https://sbert.net/docs/sentence_transformer/usage/custom_models.html#structure-of-sentence-transformer-models
    struct ggml_tensor * dense_2_out_layers   = nullptr;
    struct ggml_tensor * dense_2_out_layers_b = nullptr;
    struct ggml_tensor * dense_3_out_layers   = nullptr;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // layer -> buffer type mapping
    struct layer_buft {
        layer_buft() : buft_matrix(nullptr), buft(nullptr) {}
        layer_buft(ggml_backend_buffer_type_t matrix) : buft_matrix(matrix), buft(matrix) {}
        layer_buft(ggml_backend_buffer_type_t matrix, ggml_backend_buffer_type_t other) : buft_matrix(matrix), buft(other) {}

        ggml_backend_buffer_type_t buft_matrix; // matrices only - used by split buffers and backends that support only matrix multiplication
        ggml_backend_buffer_type_t buft;        // everything else
    };

    layer_buft buft_input;
    layer_buft buft_output;
    std::vector<layer_buft> buft_layer;

    // contexts where the model tensors metadata is stored
    std::vector<struct ggml_context *> ctxs;

    // the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_t> bufs;

    // Under hparams.no_alloc (fit's probe) every entry of `bufs` is a zero-size dummy, so the bytes the
    // weights WOULD occupy are recorded here per buffer type at load time, from
    // ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft). Empty in a normal load.
    std::map<ggml_backend_buffer_type_t, size_t> no_alloc_sizes;

    // model memory mapped files
    llama_mmaps mappings;

    // Queue readahead for the rows of `t` that a gather is about to read, in ONE batch (llama_mmap::prefetch_rows:
    // PrefetchVirtualMemory on Windows, POSIX_MADV_WILLNEED elsewhere). A no-op unless the tensor lives in one of
    // `mappings`, so a table resident in a device buffer costs nothing. Hints only; never changes results.
    // Callers: the per-layer token-embedding gathers of gemma4 and qwen4exp (llama_set_inputs).
    void prefetch_rows(const struct ggml_tensor * t, const int32_t * rows, size_t n_rows) const;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;
    // NOTE: upstream also declares `std::vector<llama_device> devices;` here (device HANDLES).
    // Excluded, not merely dropped silently: this fork already has `std::vector<int32_t> devices`
    // (device INDICES, added in the block above) with a confirmed live consumer -- src/llama.cpp
    // indexes it directly (`model.devices[i]`, passed to llama_get_device_memory /
    // llama_default_buffer_type_offload as an int device id). Two members named `devices` with
    // different element types cannot coexist; upstream's `llama_device`-handle variant belongs to
    // the same unwired buft_list-driven model-loading path already excluded in
    // llama-model-loader.h/.cpp (no confirmed caller anywhere in src/ outside that path).

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    // which tensors can be prefetched - driven by TENSOR_READ_LAZY
    std::unordered_set<const ggml_tensor *> can_prefetch;

    // statically allocated context for assigning (SPLIT_MODE_TENSOR meta device)
    struct llama_meta_device_get_split_state_userdata get_split_state_ud;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    // keep track of loaded lora adapters
    std::set<llama_lora_adapter *> lora_adapters;

    // keep track of loaded lora adapters -- upstream new-name family. Upstream carries
    // this as std::unordered_set at llama-model.h:674 and that exact type is used here so
    // the two structs do not diverge again. Freed in ~llama_model (src/llama.cpp).
    std::unordered_set<llama_adapter_lora *> loras;

    bool tensor_overrides;

    // Set by llm_apply_khad_pretransform once H is folded into wv_b/kv_b_pp.
    bool khad_pretransformed = false;

    // MADE VIRTUAL 2026-09-09 (DISPATCH_BUILD_GRAPH_2026_09_09). llama_model_base derives from
    // this struct and src/models/ defines 146 classes below it, so `delete` through a
    // `llama_model *` -- which src/llama.cpp:8473 and :8525 both do -- is undefined behaviour
    // while this destructor is non-virtual. The fork's own comment on llama_model_base's
    // destructor already named the hazard and correctly declined to fix it one level down; this
    // is the fix at the level that actually decides it.
    //
    // MEASURED SAFE before changing it, because adding a vptr changes sizeof and makes the type
    // non-standard-layout: across 774 source files under src/, common/, examples/, tools/, tests/
    // and include/, read in full, there are ZERO memcpy/memmove over a llama_model, ZERO uses of
    // sizeof(llama_model) as a byte count, ZERO offsetof into it, ZERO aggregate initialisations,
    // ZERO is_trivially_*/is_standard_layout assertions and ZERO fwrite/fread of one. Positive
    // control: `memcpy(` was found in 45 files, so the scan reached the corpus. ggml/src/** is
    // excluded because ggml never sees this type.
    virtual ~llama_model();

    // THE DISPATCHER'S GRAPH ENTRY POINT. src/llama-context.cpp calls model.build_graph(gparams)
    // at its lines 1646, 2755 and 3744; without this declaration that file cannot compile, which
    // is why the whole new-generation runtime spine is absent from the build.
    //
    // Returns nullptr here, and that is a defined answer rather than a placeholder: a plain
    // llama_model is an OLD-generation model whose graph is built by
    // llm_build_context::llama_build_graph in src/llama.cpp, and llama-context.cpp:1650 tests
    // `if (!gf)` immediately after this call. llama_model_base overrides it with upstream's body.
    virtual ggml_cgraph * build_graph(const llm_graph_params & params) const;

    // TRUE when this model's graph is built by llm_build_context::llama_build_graph in
    // src/llama.cpp rather than through build_graph() above -- i.e. exactly when build_graph()
    // returns nullptr. Named rather than inferred, because the caller that needs it cannot ask
    // build_graph(): llama_context::sched_reserve has no llm_graph_params to pass it yet, and
    // testing a nullptr return would mean building a graph to find out whether one can be built.
    // llama_model_base overrides this to false alongside its build_graph override.
    virtual bool uses_legacy_graph_builder() const { return true; }

    // THE DISPATCHER'S MEMORY ENTRY POINT (DISPATCH_CREATE_MEMORY_2026_09_09).
    // src/llama-context.cpp:615 calls memory.reset(model.create_memory(params_mem, cparams));
    // without this declaration that file cannot compile, and the whole new-generation runtime
    // spine stays out of the build.
    //
    // The body is upstream's, ported unchanged -- it chooses between llama_kv_cache,
    // llama_kv_cache_iswa, llama_memory_recurrent, llama_memory_hybrid and their DSA/MSA/DSV4
    // variants from hparams, which is precisely the decision the fork currently makes nowhere
    // because none of those classes is reachable. It is NOT virtual-dispatched per architecture:
    // upstream keeps it on the model itself and switches on `arch` inside, so this stays a single
    // definition here rather than becoming another per-class override.
    virtual llama_memory_i * create_memory(const llama_memory_params & params,
                                           const llama_cparams & cparams) const;

    size_t max_nodes(int n_tokens) const {
        auto n_tensors = tensors_by_name.size();
        if (split_mode == LLAMA_SPLIT_MODE_GRAPH && !devices.empty()) n_tensors *= devices.size();
        if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5NEXT) {
            return std::max<size_t>(n_tokens * 40, 32u * n_tensors);
        }
        //return std::max<size_t>(1024, 8*n_tensors);
        return 65536;
    }

    bool has_tensor_overrides() const {
        return tensor_overrides;
    }

    bool is_mla_model() const {
        // kimi-k3's attention layers are MLA: kv_lora_rank, attn_kv_a_mqa and the absorbed
        // wk_b/wv_b pair, with no wkv_b at all. This predicate decides the KV cache LAYOUT, so
        // omitting an MLA model here allocates it the dense shape and the MLA graph path then
        // reads a cache that was never built for it. GLM5NEXT (ik_llama #2376) is MLA as well.
        return arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_MISTRAL4 || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_GLM5NEXT;
    }

    float swiglu_limit(uint32_t il, bool shared) const {
        if (arch != LLM_ARCH_STEP35 && arch != LLM_ARCH_BAILINGMOE3 && !llm_arch_is_dsv4(arch) && arch != LLM_ARCH_GLM5NEXT && !hparams.dflash_dsv4) {
            return 0.0f;
        }
        return shared ? hparams.swiglu_clamp_shexp[il] : hparams.swiglu_clamp_exp[il];
    }

    // a compacted sliding-window cache needs the graph to build its KQ mask over the compacted
    // layout, and the compacted mask keys on position alone, so it requires a single sequence
    bool supports_dflash_swa_compress() const {
        if (!llm_arch_is_dflash_family(arch) || hparams.n_swa == 0 || hparams.n_layer == 0) {
            return false;
        }
        for (uint32_t il = 0; il < hparams.n_layer; ++il) {
            if (!hparams.is_swa_impl[il]) {
                return false;
            }
        }
        return true;
    }

    int32_t dflash_swa_compress_cross_ctx(int32_t logical_cross_ctx, bool enabled) const {
        const int32_t safe_cross_ctx = std::max<int32_t>(1, logical_cross_ctx);
        return enabled && supports_dflash_swa_compress()
                ? std::min<int32_t>(safe_cross_ctx, (int32_t) hparams.n_swa)
                : safe_cross_ctx;
    }

    bool supports_swa_compress() const {
        return arch == LLM_ARCH_OPENPANGU || llm_arch_is_dsv4(arch)
            || arch == LLM_ARCH_LAGUNA    || arch == LLM_ARCH_GEMMA4
            || arch == LLM_ARCH_GEMMA3    // fork: build_gemma3.cpp routes its SWA mask + store head like gemma4
            || supports_dflash_swa_compress() ;
    }

    static inline int hadamard_size(int head_size) {
        if ((head_size & ~(head_size - 1)) == head_size) return head_size;
        // Note: we do not include 32 as an option because the CUDA Hadamard implementation
        //       does not currently handle a block size of 32.
        for (int i = 512; i >= 64; i >>= 1) {
            if (head_size % i == 0) return i;
        }
        return 0;
    }

    inline int hadamard_size_k(int il) const {
        if (is_mla_model()) return 64;
        return hadamard_size(hparams.n_embd_head_k(il));
    }

    inline int hadamard_size_v(int il) const {
        if (is_mla_model()) return 64;
        return hadamard_size(hparams.n_embd_head_v(il));
    }

    size_t cache_size(int il, ggml_type type_k, ggml_type type_v, ggml_type idx_type_k, uint32_t kv_size, int mla_attn, int n_seq_max, bool flash_attn,
                      bool swa_compress = false, uint32_t n_ubatch = 0) const;

    void set_tensor_overrides(const llama_model_params& params);

    int device_count() const;
    ggml_backend_buffer_type_t default_buffer_type_offload(int device) const;

    // S69-WIRE66-CONTEXT-MEMORY-ARCH-DECISION step 1: real per-layer device accessor, backed by
    // the existing `buft_layer` mapping the loader already populates (llama.cpp, model load path)
    // -- NOT a stub. Mirrors upstream llama_model::dev_layer(il) call shape so the unbuilt memory
    // TUs (llama-kv-cache-dsv4.cpp, llama-memory-recurrent.cpp, llama-kv-cache.cpp) compile
    // against a genuine mapping instead of a fabricated one. Out-of-range `il` (>= buft_layer.size())
    // returns nullptr, same as an out-of-range buft_layer[il] access anywhere else this member is
    // already read in the built loader (src/llama.cpp) -- this accessor does not add a new null
    // risk, it exposes the existing buft_layer contract faithfully.
    ggml_backend_dev_t dev_layer(uint32_t il) const;

    // ADDED 2026-09-09, slice 16a. Three accessors upstream's llama-context.cpp calls on
    // llama_model, each backed by data this fork ALREADY loads. Same shape and same reasoning as
    // dev_layer() above: mirror upstream's CALL SHAPE over the fork's own data, rather than
    // importing upstream's implementation or its data layout.
    //
    // n_devices() is deliberately NOT among them: this fork already spells that device_count(),
    // so the call site is rewritten instead. Two names for one fact is how several apparent
    // "losses" in this merge turned out to be renames.
    std::string arch_name() const;
    size_t n_tensors() const;
    ggml_backend_dev_t dev_output() const;

    // ADDED 2026-09-09, slice 25. Per-buffer-type memory used by this model, summed over the
    // buffers the loader actually allocated. llama_context::memory_breakdown() asks for this and
    // upstream declares the same signature (279b4bfaf^2:src/llama-model.h:700); this fork never had
    // it. Same shape as arch_name/n_tensors/dev_output above -- a read-only accessor over state the
    // model already owns, not new state.
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // ADDED 2026-09-09, slice 30. Per-layer RoPE frequency selection, ported from the fork's own
    // live logic rather than stubbed -- src/llama-build-context.cpp:527-539 (build_rope_factors)
    // and :3279-3282 (build_std_attention's freq_base/freq_scale choice plus the per-layer
    // override). Declared by upstream with these exact signatures and called from
    // src/llama-kv-cache.cpp:1993-1996 and from ~100 files under src/models/, none of which could
    // compile without them.
    //
    // READ THIS BEFORE USING THEM ON A NEW ARCHITECTURE. The SWA arms key on hparams.is_swa(il),
    // which reads the is_swa_impl per-layer array. That is upstream's predicate and the one
    // src/models/ is written against -- but this tree holds THREE representations of "is layer il
    // sliding": is_swa_impl (here), hparams.n_swa_pattern plus a modular formula
    // (src/graphs/build_llama.cpp:44), and hardcoded local constants
    // (src/graphs/build_cohere2.cpp:26). A loader that populates only one of the others leaves
    // is_swa_impl all-zero, and these accessors then return the DENSE values for layers the graph
    // builder treats as sliding -- silently, with no compile error and no runtime complaint.
    // MEASURED INSTANCE: the COHERE2 loader (src/llama-hparams.cpp:1549) sets n_swa_pattern = 4 and
    // n_swa but never writes is_swa_impl. It does not bite today only because upstream's
    // src/models/cohere2.cpp has both frequency accessors commented out as UNUSED.
    //
    // The non-SWA arms read CPARAMS, not hparams, and that is load-bearing: src/llama.cpp:8831
    // derives cparams.rope_freq_scale from hparams.rope_freq_scale_train while honouring the user's
    // --rope-freq-scale, and :8868 forces it to 1.0f when the scaling type is NONE. Reading hparams
    // here would discard both.
    float         get_rope_freq_base (const llama_cparams & cparams, int il) const;
    float         get_rope_freq_scale(const llama_cparams & cparams, int il) const;
    ggml_tensor * get_rope_factors   (const llama_cparams & cparams, int il) const;

    std::vector<float> splits;
    ggml_backend_buffer_type_t split_buft = nullptr;

    std::unique_ptr<reload_info> reload;
};

// Recompute the load-time-derived combined wkv_b (computed_wkv_b) of layer il
// from the current wk_b/wv_b tensor data (hot-swap support). Defined in
// llama.cpp next to llm_prepare_mla.
bool llm_refresh_computed_wkv_b(llama_model & model, int il);

struct llama_lora_weight {
    struct ggml_tensor * a = nullptr;
    struct ggml_tensor * b = nullptr;
    llama_lora_weight() = default;
    llama_lora_weight(struct ggml_tensor * a, struct ggml_tensor * b): a(a), b(b) {}
};

struct llama_lora_adapter {
    llama_model * base_model;
    // map tensor name to lora_a_b
    std::unordered_map<std::string, struct llama_lora_weight> ab_map;
    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    float alpha;

    llama_lora_adapter(struct llama_model * base_model): base_model(base_model) {
        base_model->lora_adapters.insert(this);
    }

    llama_lora_weight * get_weight(struct ggml_tensor * w) {
        std::string name(w->name);
        auto pos = ab_map.find(name);
        if (ab_map.find(name) != ab_map.end()) {
            return &pos->second;
        }
        return nullptr;
    }

    ~llama_lora_adapter() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
        auto pos = base_model->lora_adapters.find(this);
        if (pos != base_model->lora_adapters.end()) {
            base_model->lora_adapters.erase(pos);
        }
    }
};

// helper to handle gguf constants
// usage:
//
//   const auto tn = LLM_TN(LLM_ARCH_LLAMA);
//
//   std::string name = tn(LLM_TENSOR_OUTPUT);                     -> "output"
//   std::string name = tn(LLM_TENSOR_TOKEN_EMBD, "bias");         -> "token_embd.bias"
//   std::string name = tn(LLM_TENSOR_ATTN_NORM, "weight", 3);     -> "blk.3.attn_norm.weight"
//
// what LLM_TN returns: a std::string that also answers upstream's `.str()` (upstream's tn() returns LLM_TN_IMPL), so the
// upstream model files and this fork's callers both compile unchanged (merge 2026-11-04)
struct llm_tn_name : std::string {
    llm_tn_name(std::string s) : std::string(std::move(s)) {}
    const std::string & str() const { return *this; }
};

struct LLM_TN {
    LLM_TN(llm_arch arch, bool generic_first = false) : arch(arch), generic_first(generic_first) {}

    llm_arch arch;
    // Resolve through upstream's generic name table BEFORE the arch's legacy table. Set only for a class-built model whose
    // arch also has a legacy loader with different names: the legacy LLM_ARCH_DFLASH table maps ATTN_POST_NORM to
    // "blk.%d.ffn_norm", so upstream's llama_model_dflash (FFN_NORM + optional ATTN_POST_NORM) created that one file tensor
    // twice per layer ("expected 103, got 108; created more than once: blk.0..4.ffn_norm.weight", LiLiCorr draft 2026-11-05).
    bool generic_first;

    llm_tn_name operator()(llm_tensor tensor) const;

    llm_tn_name operator()(llm_tensor tensor, const std::string & suffix) const;

    llm_tn_name operator()(llm_tensor tensor, int bid) const;

    llm_tn_name operator()(llm_tensor tensor, const std::string & suffix, int bid) const;

    llm_tn_name operator()(llm_tensor tensor, const std::string & suffix, int bid, int xid) const;
};

std::string llama_model_ftype_name(llama_ftype ftype);

const char * llama_model_type_name(e_model type);

// INVENT-LADDER-OK: full ladder already run by the survey this implements (existing fix: none,
// checked; prior art: upstream's own two-level hierarchy IS the candidate, adapted as option (ii)
// below rather than adopted whole per the survey's own cost/risk comparison in section D; adapt:
// llama_model_base bridges the fork's UNCHANGED concrete llama_model instead of replacing it).
// The two items flagged below as "not fixed here" (non-virtual dtor, per-buft ctx routing) are
// NAMED, SCOPED, DELIBERATE deferrals to option (i) -- costed and sequenced in the survey's own
// section D/QUEUED-1, not silent abandonment: option (i) is recorded as the correct end state,
// option (ii) as the immediate low-risk unblocker, exactly as the task's own DoD requires ("What
// this option does NOT do" is itself a required, stated line item, not a gap left unnoticed).
//
// EXCLUDED, not silently: upstream's side of this block declares `struct llama_model` as an
// ABSTRACT POLYMORPHIC BASE CLASS (pure-virtual load_tensors/build_graph/etc, one llama_model_base
// subclass per architecture) -- a fundamentally different design for the SAME type name than
// this fork's plain data struct above, so the two cannot coexist (redefinition). MEASURED which
// design is actually live: src/llama.cpp's llama_model_load() (the real body behind the public
// llama_model_load_from_file) operates on `llama_model & model` as a plain struct via free
// functions (llm_load_arch/llm_load_hparams), never through a virtual call; src/llama-load-tensors.cpp
// (create_tensors_helper, the live tensor-creation path) likewise takes `llama_model & model` by
// plain reference. Upstream's llama_model_base::create_tensor(hparams, buft_lists...) has no
// resolved caller anywhere in src/ -- the same finding already recorded in
// llama-model-loader.h/.cpp for the parallel buft_list-driven create_tensor overload this class
// hierarchy exists to serve.
//
// SUPERSEDED BELOW (2026-09-08, S69-SRC-MODELS-BLOCKED-BY-MISSING-LLAMA-MODEL-BASE-HIERARCHY-
// 2026-09-08-01, option (ii)). RESEARCH-DONE: temp/research/findings/
// S69_LLAMA_MODEL_MIGRATION_SURVEY_2026-09-08.md, section QUEUED-2, read in full (640 lines) --
// quote: "The note's literal text ... holds exactly as measured for the built set (B: 0 built
// includers of models.h, confirmed by a full CMakeLists.txt read plus a whole-src #include
// sweep). This survey's own measurement does not conflict with that claim. What this survey
// adds ... is the whole-tree number the note's phrasing does not separately surface: 150 files /
// 146 classes / 139 direct-`llama_model_base` subclasses exist on disk, unbuilt". So:
// llm_type_name(), the LLAMA_LOAD_LOCALS macro, and llama_model_base were excluded above "zero
// confirmed callers" for the BUILT tree -- true, and still true (survey section B: 0 built
// includers of models.h, exact quote above). But the 150 UNBUILT files in src/models/*.cpp are
// on disk, needing exactly this hierarchy to compile -- the whole-tree number the original note's
// own scope did not separately state. The bridge below adds llama_model_base back as a NEW class
// over the fork's UNCHANGED concrete llama_model (does not revive upstream's abstract-llama_model
// design this note correctly excluded), plus the LLM_TYPE_*/MODEL_* enum bridge and
// LLAMA_LOAD_LOCALS macro those 150 files also need. llama_internal_get_tensor_map is unaffected
// either way: kept, because src/llama.cpp (already resolved) DEFINES it independent of this class
// hierarchy, so the declaration is required for that definition to have a prototype in scope.
#define LLM_TYPE_UNKNOWN MODEL_UNKNOWN
#define LLM_TYPE_14M MODEL_14M
#define LLM_TYPE_17M MODEL_17M
#define LLM_TYPE_22M MODEL_22M
#define LLM_TYPE_33M MODEL_33M
#define LLM_TYPE_60M MODEL_60M
#define LLM_TYPE_70M MODEL_70M
#define LLM_TYPE_80M MODEL_80M
#define LLM_TYPE_109M MODEL_109M
#define LLM_TYPE_137M MODEL_137M
#define LLM_TYPE_140M MODEL_140M
#define LLM_TYPE_160M MODEL_160M
#define LLM_TYPE_190M MODEL_190M
#define LLM_TYPE_220M MODEL_220M
#define LLM_TYPE_250M MODEL_250M
#define LLM_TYPE_256M MODEL_256M
#define LLM_TYPE_270M MODEL_270M
#define LLM_TYPE_335M MODEL_335M
#define LLM_TYPE_350M MODEL_350M
#define LLM_TYPE_360M MODEL_360M
#define LLM_TYPE_410M MODEL_410M
#define LLM_TYPE_450M MODEL_450M
#define LLM_TYPE_475M MODEL_475M
#define LLM_TYPE_558M MODEL_558M
#define LLM_TYPE_700M MODEL_700M
#define LLM_TYPE_770M MODEL_770M
#define LLM_TYPE_780M MODEL_780M
#define LLM_TYPE_950M MODEL_950M
#define LLM_TYPE_0_3B MODEL_0_3B
#define LLM_TYPE_0_5B MODEL_0_5B
#define LLM_TYPE_0_6B MODEL_0_6B
#define LLM_TYPE_0_8B MODEL_0_8B
#define LLM_TYPE_1B MODEL_1B
#define LLM_TYPE_1_2B MODEL_1_2B
#define LLM_TYPE_1_3B MODEL_1_3B
#define LLM_TYPE_1_4B MODEL_1_4B
#define LLM_TYPE_1_5B MODEL_1_5B
#define LLM_TYPE_1_6B MODEL_1_6B
#define LLM_TYPE_1_7B MODEL_1_7B
#define LLM_TYPE_1_8B MODEL_1_8B
#define LLM_TYPE_2B MODEL_2B
#define LLM_TYPE_2_6B MODEL_2_6B
#define LLM_TYPE_2_8B MODEL_2_8B
#define LLM_TYPE_2_9B MODEL_2_9B
#define LLM_TYPE_3B MODEL_3B
#define LLM_TYPE_4B MODEL_4B
#define LLM_TYPE_6B MODEL_6B
#define LLM_TYPE_6_9B MODEL_6_9B
#define LLM_TYPE_7B MODEL_7B
#define LLM_TYPE_8B MODEL_8B
#define LLM_TYPE_9B MODEL_9B
#define LLM_TYPE_11B MODEL_11B
#define LLM_TYPE_12B MODEL_12B
#define LLM_TYPE_13B MODEL_13B
#define LLM_TYPE_14B MODEL_14B
#define LLM_TYPE_15B MODEL_15B
#define LLM_TYPE_16B MODEL_16B
#define LLM_TYPE_20B MODEL_20B
#define LLM_TYPE_26B_A4B MODEL_26B_A4B
#define LLM_TYPE_27B MODEL_27B
#define LLM_TYPE_30B MODEL_30B
#define LLM_TYPE_32B MODEL_32B
#define LLM_TYPE_34B MODEL_34B
#define LLM_TYPE_35B MODEL_35B
#define LLM_TYPE_36B MODEL_36B
#define LLM_TYPE_40B MODEL_40B
#define LLM_TYPE_65B MODEL_65B
#define LLM_TYPE_70B MODEL_70B
#define LLM_TYPE_120B MODEL_120B
#define LLM_TYPE_142B MODEL_142B
#define LLM_TYPE_236B MODEL_236B
#define LLM_TYPE_290B MODEL_290B
#define LLM_TYPE_314B MODEL_314B
#define LLM_TYPE_405B MODEL_405B
#define LLM_TYPE_671B MODEL_671B
#define LLM_TYPE_SMALL MODEL_SMALL
#define LLM_TYPE_MEDIUM MODEL_MEDIUM
#define LLM_TYPE_LARGE MODEL_LARGE
#define LLM_TYPE_XL MODEL_XL
#define LLM_TYPE_A1_7B MODEL_A1_7B
#define LLM_TYPE_A2_7B MODEL_A2_7B
#define LLM_TYPE_8x7B MODEL_8x7B
#define LLM_TYPE_8x22B MODEL_8x22B
#define LLM_TYPE_16x12B MODEL_16x12B
#define LLM_TYPE_16x3_8B MODEL_16x3_8B
#define LLM_TYPE_10B_128x3_66B MODEL_10B_128x3_66B
#define LLM_TYPE_57B_A14B MODEL_57B_A14B
#define LLM_TYPE_17B_16E MODEL_17B_16E
#define LLM_TYPE_17B_128E MODEL_17B_128E
#define LLM_TYPE_A13B MODEL_A13B
#define LLM_TYPE_7B_A1B MODEL_7B_A1B
#define LLM_TYPE_8B_A1B MODEL_8B_A1B
#define LLM_TYPE_12B_A2_5B MODEL_12B_A2_5B
#define LLM_TYPE_16B_A1B MODEL_16B_A1B
#define LLM_TYPE_21B_A3B MODEL_21B_A3B
#define LLM_TYPE_30B_A3B MODEL_30B_A3B
#define LLM_TYPE_33B_A3B MODEL_33B_A3B
#define LLM_TYPE_35B_A3B MODEL_35B_A3B
#define LLM_TYPE_80B_A3B MODEL_80B_A3B
#define LLM_TYPE_80B_A13B MODEL_80B_A13B
#define LLM_TYPE_100B_A6B MODEL_100B_A6B
#define LLM_TYPE_106B_A12B MODEL_106B_A12B
#define LLM_TYPE_119B_A6B MODEL_119B_A6B
#define LLM_TYPE_122B_A10B MODEL_122B_A10B
#define LLM_TYPE_230B_A10B MODEL_230B_A10B
#define LLM_TYPE_235B_A22B MODEL_235B_A22B
#define LLM_TYPE_310B_A15B MODEL_310B_A15B
#define LLM_TYPE_300B_A47B MODEL_300B_A47B
#define LLM_TYPE_355B_A32B MODEL_355B_A32B
#define LLM_TYPE_397B_A17B MODEL_397B_A17B
#define LLM_TYPE_177B_A58B MODEL_177B_A58B
#define LLM_TYPE_744B_A40B MODEL_744B_A40B
#define LLM_TYPE_E2B MODEL_E2B
#define LLM_TYPE_E4B MODEL_E4B
#define LLM_TYPE_47M MODEL_47M
#define LLM_TYPE_149M MODEL_149M
#define LLM_TYPE_230M MODEL_230M
#define LLM_TYPE_395M MODEL_395M
#define LLM_TYPE_26B MODEL_26B
#define LLM_TYPE_31B MODEL_31B
#define LLM_TYPE_456B MODEL_456B
#define LLM_TYPE_7_9B_A1_3B MODEL_7_9B_A1_3B
#define LLM_TYPE_24B_A2B MODEL_24B_A2B
#define LLM_TYPE_31B_A3_5B MODEL_31B_A3_5B
#define LLM_TYPE_48B_A3B MODEL_48B_A3B
#define LLM_TYPE_102B_A12B MODEL_102B_A12B
#define LLM_TYPE_118B_A8B MODEL_118B_A8B
#define LLM_TYPE_120B_A12B MODEL_120B_A12B
#define LLM_TYPE_124B_A5_1B MODEL_124B_A5_1B
#define LLM_TYPE_196B_A11B MODEL_196B_A11B
#define LLM_TYPE_428B_A23B MODEL_428B_A23B
#define LLM_TYPE_288B_A19B MODEL_288B_A19B
#define LLM_TYPE_685B_A37B MODEL_685B_A37B
#define LLM_TYPE_2_8T_A50B MODEL_2_8T_A50B
#define LLM_TYPE_816B_A42B MODEL_816B_A42B
#define LLM_TYPE_78B_A3_5B MODEL_78B_A3_5B
#define LLM_TYPE_1B_A400M MODEL_1B_A400M
#define LLM_TYPE_3B_A800M MODEL_3B_A800M
#define LLM_TYPE_32B_A9B MODEL_32B_A9B
#define LLM_TYPE_75B_A9B MODEL_75B_A9B
#define LLM_TYPE_A3B MODEL_A3B
#define LLM_TYPE_320B_A18B MODEL_320B_A18B

// ADOPTED from upstream (commit 994118a18 "model: move load_hparams and load_tensors to
// per-model definition") in this merge, ADAPTED for this fork's hparams shape: every
// src/models/*.cpp file's load_arch_tensors()/graph body expects these bare local names in
// scope (confirmed by the talkie.cpp compile probe: n_layer/n_head/n_embd/n_vocab/n_ff/
// n_embd_gqa/n_embd_head_k all "not declared in this scope" before this macro exists). ONE
// REAL DIFFERENCE from upstream's macro, not a copy-paste: upstream calls `hparams.n_rot()` as
// a METHOD; this fork's llama_hparams (llama-hparams.h) deliberately keeps `n_rot` as a PLAIN
// FIELD (see llama-hparams.h's own comment on why n_layer() could not be reused there for the
// same reason) with no `n_rot()` method overload at all (only `rope_n_rot(il)`, a different,
// SWA-aware accessor) -- calling `hparams.n_rot()` here would fail to compile with "uint32_t
// value is not a function". Adapted to `hparams.n_rot` (field, no call) instead; every other
// line matches upstream's macro exactly, name for name.
// CORRECTED 2026-09-09. `n_layer` here USED to be hparams.n_layer -- this fork's TOTAL layer
// count -- and every one of this macro's 138 users is a file from src/models/, which came from
// upstream verbatim (150 of 150 present at the merge parent 279b4bfaf^2; 0 fork-authored). In
// upstream, the local `n_layer` means the EFFECTIVE count and `n_layer_all` means the total. So
// this macro was handing 138 upstream files the wrong one of two numbers that differ ONLY on
// models with nextn/MTP layers -- silently, and only on those models, which is why it went unseen.
//
// n_layer_all was missing entirely, and supplying it as hparams.n_layer while leaving n_layer as
// the total would have been worse than the error it removes: src/models/bailingmoe3.cpp:128 and
// cohere2moe.cpp:148 both write `for (il = n_layer; il < n_layer_all; ++il)`, a loop over exactly
// the MTP layers, which would collapse to an EMPTY loop and quietly skip loading a model's draft
// head with no diagnostic at all.
//
// Safe to change because the macro has ZERO users outside src/models/ (case-checked), and
// n_layer_eff() is upstream's n_layer() -- see the note at its definition, which cites
// 279b4bfaf^2:src/llama-hparams.cpp:301, `n_layer_all - n_layer_nextn`.
#define LLAMA_LOAD_LOCALS \
    const int     n_layer        = (int) hparams.n_layer_eff(); GGML_UNUSED(n_layer); \
    const int     n_layer_all    = (int) hparams.n_layer;    GGML_UNUSED(n_layer_all); \
    /* ADDED 2026-09-09 (slice 49). Two models sources -- deepseek4.cpp:97 and \
       gemma4-assistant.cpp:52 -- use this name in real code and failed with \
       "'n_layer_nextn' was not declared in this scope". The macro exists to give each \
       models/ TU the local names upstream's monolithic loader had; this one was \
       missing. Bound as int, like n_layer and n_layer_all above, because \
       gemma4-assistant.cpp writes `for (int i = 0; i < n_layer_nextn; ++i)`. */ \
    const int     n_layer_nextn  = (int) hparams.n_layer_nextn; GGML_UNUSED(n_layer_nextn); \
    const int64_t n_head         = hparams.n_head();         GGML_UNUSED(n_head); \
    const int64_t n_head_kv      = hparams.n_head_kv();      GGML_UNUSED(n_head_kv); \
    const int64_t n_embd         = hparams.n_embd;           GGML_UNUSED(n_embd); \
    const int64_t n_embd_k_gqa   = hparams.n_embd_k_gqa();   GGML_UNUSED(n_embd_k_gqa); \
    const int64_t n_embd_v_gqa   = hparams.n_embd_v_gqa();   GGML_UNUSED(n_embd_v_gqa); \
    const int64_t n_embd_head_k  = hparams.n_embd_head_k();  GGML_UNUSED(n_embd_head_k); \
    const int64_t n_embd_head_v  = hparams.n_embd_head_v();  GGML_UNUSED(n_embd_head_v); \
    const int64_t n_ff           = hparams.n_ff();           GGML_UNUSED(n_ff); \
    const int64_t n_embd_gqa     = n_embd_v_gqa;             GGML_UNUSED(n_embd_gqa); \
    const int64_t n_vocab        = vocab.n_tokens();         GGML_UNUSED(n_vocab); \
    const int64_t n_token_types  = vocab.n_token_types();    GGML_UNUSED(n_token_types); \
    const int64_t n_rot          = hparams.n_rot;            GGML_UNUSED(n_rot); \
    const int64_t n_expert       = hparams.n_expert;         GGML_UNUSED(n_expert); \
    const int64_t n_expert_used  = hparams.n_expert_used_at();  GGML_UNUSED(n_expert_used); \
    const int64_t n_ctx_train    = hparams.n_ctx_train;      GGML_UNUSED(n_ctx_train);

// model must inherit from this (upstream's own comment, kept: every one of the 139 classes in
// models.h does `struct llama_model_X : public llama_model_base { llama_model_X(params) :
// llama_model_base(params) {} ... }`, confirmed the ONLY constructor shape all 139 use).
//
// Everything below is defined INLINE in this header, on purpose: this class is not added to
// src/CMakeLists.txt by this change (that glob is a separate decision, out of this task's
// scope -- see the task's own WHAT NOT TO DO), so there is no llama-model-base.cpp translation
// unit for out-of-line bodies to live in and link against. Inline bodies compile standalone in
// whichever src/models/*.cpp TU includes this header, with no ODR risk from multiple TUs (the
// usual reason to prefer out-of-line) because nothing links these TUs together yet either.
// The architecture the next llama_model_base is being created for. The base builds its LLM_TN in its constructor, i.e. BEFORE the loader has set `arch`, and all
// 146 derived classes call `llama_model_base(params)` with no arch argument, so the factory (llama-model-create.cpp) leaves it here for the constructor to read.
inline llm_arch & llama_model_pending_arch() {
    static thread_local llm_arch a = LLM_ARCH_UNKNOWN;
    return a;
}

// Creates the src/models class for an architecture that has no legacy loader, or returns nullptr for every other architecture (the caller then makes a plain llama_model).
llama_model * llama_model_create_for_arch(llm_arch arch, const llama_model_params & params);
// The src/models DFlash class for a LiLiCorr draft (an LLM_ARCH_DFLASH GGUF with lilicorr.* keys); see llama-model-create.cpp.
llama_model * llama_model_create_dflash_class(const llama_model_params & params);
bool llama_arch_is_class_dispatched(llm_arch arch);

struct llama_model_base : public llama_model {
    llama_model * model;
    llama_model_loader * ml = nullptr;
    const LLM_TN tn;

    // Routes a class loader's create_tensor() through the live loader's per-layer context choice (create_tensors_helper). Unset = the single-context fallback below.
    std::function<ggml_tensor * (const std::string &, const std::vector<int64_t> &, int)> create_tensor_router;

    // llama_model_loader is fully defined by the include added near the top of this file (was:
    // forward-declared only), so these can copy the real values at construction time instead of
    // duplicating the bit patterns by hand.
    const int TENSOR_DUPLICATED;
    const int TENSOR_NOT_REQUIRED;
    const int TENSOR_SKIP;
    const int TENSOR_SKIP_IF_VIRTUAL;
    const int TENSOR_ALLOW_RESHAPE;
    const int TENSOR_READ_LAZY; // read rows on demand (upstream; loader flag 1 << 5)

    // Step 2 of option (ii): a no-op base-class pass-through. The fork's llama_model (above) has
    // no declared constructor of its own -- it is default-/aggregate-constructed today and filled
    // in afterward by free functions (llm_load_arch/llm_load_hparams/create_tensors_helper) -- so
    // there is nothing for this constructor to forward `params` INTO yet. `params` is accepted
    // (every models.h subclass's initializer list requires the signature to match) and otherwise
    // unused until option (i) gives llama_model_base real load-time responsibilities.
    explicit llama_model_base(const llama_model_params & params)
        : model(this), tn(llama_model_pending_arch(), llama_model_pending_arch() == LLM_ARCH_DFLASH),
          TENSOR_DUPLICATED     (llama_model_loader::TENSOR_DUPLICATED),
          TENSOR_NOT_REQUIRED   (llama_model_loader::TENSOR_NOT_REQUIRED),
          TENSOR_SKIP           (llama_model_loader::TENSOR_SKIP),
          TENSOR_SKIP_IF_VIRTUAL(llama_model_loader::TENSOR_SKIP_IF_VIRTUAL),
          TENSOR_ALLOW_RESHAPE  (llama_model_loader::TENSOR_ALLOW_RESHAPE),
          TENSOR_READ_LAZY      (llama_model_loader::TENSOR_READ_LAZY) {
        // ADDED 2026-09-09, and it is a defect fix rather than tidying. `mtp` is declared at
        // llama_model:961 with NO initialiser, and until this line `params` was accepted and
        // discarded (`GGML_UNUSED(params)`), so on every llama_model_base the flag was
        // INDETERMINATE -- not false, indeterminate. The merged src/models/ tensor loaders decide
        // whether to load a model's MTP tensors from it, so reading it was undefined behaviour
        // whose visible symptom would be "MTP sometimes loads".
        //
        // It reads params.mtp, which is the switch this fork actually drives (src/llama.cpp:5114
        // does `model.mtp = params.mtp` on the live path). It deliberately does NOT read
        // params.load_mtp -- see the note at that field in include/llama.h: that one is upstream's
        // spelling, adopted by this merge, and read by nothing.
        mtp = params.mtp;
        // `arch` is set again by the loader (llm_load_arch) from the file; setting it here keeps tn and arch in step if anything reads it earlier
        arch = tn.arch;
    }

    // NON-virtual on purpose, matching TRAP 1 above: llama_model's own dtor is non-virtual, so
    // giving THIS class a virtual dtor would not fix deleting through a `llama_model *` (the
    // hazard TRAP 1 names); it would only mask it one level down. Left non-virtual, consistent
    // with the fork's own choice, not silently "improved" into a different, still-incomplete fix.
    ~llama_model_base() = default;

    // ADDED 2026-09-09 (DISPATCH_BUILD_GRAPH_2026_09_09). The comment above is about the
    // DESTRUCTOR and it still reads correctly -- it is left untouched -- but its premise has
    // changed: llama_model's destructor is now virtual, so this defaulted one is implicitly
    // virtual too and deleting through a `llama_model *` is well-defined at last.
    //
    // This override carries upstream's real llama_model::build_graph body (adapted: upstream
    // declares build_arch_graph on llama_model itself, which is an abstract base there; in this
    // fork llama_model is a plain data struct and build_arch_graph is declared here, so the body
    // belongs at this level). It is the single point where an architecture's own graph builder is
    // finally reached from the runtime.
    // upstream helpers called by newly merged src/models classes (bodies in llama-model.cpp)
    // tensor flags for a file that holds the trunk, the NextN layers, or both
    struct nextn_flags_t {
        int trunk; // TENSOR_NOT_REQUIRED when the file holds only the NextN layers
        int mtp;   // TENSOR_NOT_REQUIRED when the file holds only the trunk, TENSOR_SKIP when MTP is not loaded
    };
    nextn_flags_t nextn_flags(llama_model_loader & ml, llm_tensor trunk_probe = LLM_TENSOR_ATTN_NORM) const;

    // read the SWA pattern as one flag per layer, or as a period expanded by set_swa_pattern
    void load_swa_pattern(llama_model_loader & ml, uint32_t n_pattern, bool dense_first = false);

    ggml_cgraph * build_graph(const llm_graph_params & params) const override;

    // Overridden alongside build_graph: a new-generation model builds through that virtual, not
    // through llm_build_context::llama_build_graph. Kept adjacent to the override it tracks so the
    // two cannot drift apart.
    bool uses_legacy_graph_builder() const override { return false; }

    // Survey section C: SAME PURPOSE as upstream's create_tensor(ml, tn, ne, flags), DIFFERENT
    // SHAPE, MUST be adapted rather than merely renamed. Upstream resolves its ggml_context via
    // pimpl->dev_layer/pimpl->cpu_buft_list (upstream-only state, confirmed absent from this fork
    // by reading src/llama-model.h in full, section A). This fork's equivalent per-buft ctx_map
    // routing lives inside the LIVE create_tensors_helper (src/llama-load-tensors.cpp) as
    // per-load-pass state that is never persisted onto llama_model itself, so it cannot be
    // reached from a method defined here. What CAN be reached: `ctxs`, the
    // `std::vector<struct ggml_context *>` llama_model already carries (declared above, "contexts
    // where the model tensors metadata is stored") and the live loader already populates during a
    // real load. This picks the first entry as a real, correct-for-a-single-context answer and
    // forwards to the loader's own already-built create_tensor(ctx, name, ne, flags) -- a genuine
    // adaptation, not a stub, but it does NOT replicate the live loader's per-buft/tensor-parallel
    // routing; that reconciliation is exactly option (i)'s still-owed work (section D), named here
    // rather than silently assumed done.
    ggml_tensor * create_tensor(llama_model_loader & ml_, const std::string & name, const std::vector<int64_t> & ne, int flags = 0) {
        struct ggml_context * ctx = ctxs.empty() ? nullptr : ctxs.front();
        GGML_ASSERT(ctx != nullptr && "llama_model_base::create_tensor: model has no ggml_context (ctxs empty) -- not wired into a real load pass, see llama-model.h's llama_model_base comment");
        return ml_.create_tensor(ctx, name, ne, flags);
    }

    // convenience overload of create_tensor that doesn't require an explicit llama_model_loader
    // (matches every models/*.cpp call site, e.g. talkie.cpp: `create_tensor(tn(...), {..}, 0)`).
    // Survey section C confirmed LLM_TN_IMPL has zero live callers outside two .pre-* backup
    // files and the fork's own live LLM_TN::operator() already returns std::string directly, so
    // this is typed `const std::string &` rather than reviving that proxy type.
    ggml_tensor * create_tensor(const std::string & name, const std::vector<int64_t> & ne, int flags = 0) {
        if (create_tensor_router) {
            return create_tensor_router(name, ne, flags);
        }
        GGML_ASSERT(ml != nullptr && "llama_model_base::create_tensor: no llama_model_loader set (this->ml) -- not wired into a real load pass, see llama_model_base's comment");
        return create_tensor(*ml, name, ne, flags);
    }

    // ADOPTED verbatim from upstream's real body (git history: commit 279b4bfaf's upstream side,
    // src/llama-model.cpp's llama_model_base::create_tensor_gate_up_exps) -- confirmed absent as
    // a reusable helper on this fork's side (survey section C: create_tensors_helper's per-arch
    // methods do this fallback inline, not through one shared function), so this is real,
    // net-new-to-this-fork logic, not a rename. Field names (ffn_gate_up_exps/ffn_gate_exps/
    // ffn_up_exps) confirmed present on this fork's llama_layer, same names, same types.
    void create_tensor_gate_up_exps(llama_layer & layer, int bid, int64_t n_embd_, int64_t n_ff_, int64_t n_expert_, int flags) {
        // SKIP_IF_VIRTUAL: a metadata-only model must not conjure the fused alternative (cohere2moe, hy_v3: its split views fail ggml_is_contiguous_1 when n_expert_used == 1)
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
        if (layer.ffn_gate_up_exps == nullptr) {
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
        }
    }

    // ADOPTED verbatim from upstream's real body (same commit/file as above,
    // llama_model_base::create_tensor_qkv) -- confirmed absent as a reusable helper on this
    // fork's side (survey section C: create_tensors_helper calls create_tensor_qkv directly in
    // src/models/*.cpp bodies already, confirming the NAME is expected to be inherited, but no
    // standalone method of that name exists on the fork's own helper). Field names (wqkv/wqkv_b/
    // wq/wk/wv/wq_b/wk_b/wv_b) confirmed present on this fork's llama_layer, same names, same
    // types (llama-model.h:323-339).
    void create_tensor_qkv(llama_layer & layer, int bid,
            int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
            int flags) {
        const int64_t n_embd_qkv = n_embd_q_ + n_embd_k_ + n_embd_v_;

        if (flags & TENSOR_SKIP) {
            const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;

            create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, skip | TENSOR_SKIP_IF_VIRTUAL);
            create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   bid), {n_embd_qkv},          skip | TENSOR_SKIP_IF_VIRTUAL);
            create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", bid), {n_embd_, n_embd_q_},  skip);
            create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", bid), {n_embd_, n_embd_k_},  skip);
            create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", bid), {n_embd_, n_embd_v_},  skip);
            create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias",   bid), {n_embd_q_},           skip);
            create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias",   bid), {n_embd_k_},           skip);
            create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias",   bid), {n_embd_v_},           skip);
            return;
        }

        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
        if (layer.wqkv) {
            layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", bid), {n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", bid), {n_embd_, n_embd_q_}, flags);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", bid), {n_embd_, n_embd_k_}, flags);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", bid), {n_embd_, n_embd_v_}, flags);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
        }
    }

    // Step 5 of option (ii), stated in code, not just in the survey: PURE VIRTUAL, matching
    // upstream exactly, and the fork's 3 live switch-dispatch functions (llm_load_hparams,
    // create_tensors_helper::create_tensors, llm_build_context::llama_build_graph) are NOT
    // retired and NOT called from here. Each of the 139 classes in models.h already overrides
    // all three (that is the 150-file surface this whole change exists to unblock).
    virtual void load_arch_hparams(llama_model_loader & ml) = 0;
    virtual void load_arch_tensors(llama_model_loader & ml) = 0;
    virtual std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const = 0;
};

// llama_internal_get_tensor_map is kept (as before this change): src/llama.cpp (already
// resolved) DEFINES it independent of this class hierarchy, so the declaration is required for
// that definition to have a prototype in scope.
const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);
