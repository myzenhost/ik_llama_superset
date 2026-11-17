#pragma once

#include "llama-impl.h"

#include <cstdint>
#include <string>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  snapkv_cap;      // SnapKV prompt-KV cap (0=off) — S69 #22
    int32_t  snapkv_window;   // SnapKV observation window
    int32_t  h2o_heavy;       // H2O heavy-hitter budget (0=off)
    int32_t  h2o_recent;      // H2O recent window kept
    int32_t  kvflash_pool_tokens; // KVFlash paged-KV resident pool in tokens (0=off) — S69 #66
    bool     offload_attn;          // S69 #220 offload-attention (default false)
    int32_t  offload_attn_page;     // tokens per KV page
    float    offload_attn_budget;   // fraction of pages attended per step
    int32_t  offload_attn_selector; // 0 quest, 1 shadowkv, 2 retattn
    int32_t  offload_attn_sink;     // leading KV pages pinned regardless of score
    int32_t  offload_attn_recent;   // recent local window (pages ending at the query's page) kept in addition to the top-k; both graph paths
    int32_t  offload_attn_dense_layers; // the first N layers are never masked (attend densely); 0 = mask every layer
    int32_t  offload_attn_refresh;  // FA path: reuse the page selection for R single-token decode steps (1 = rescan every step)
    int32_t  offload_attn_scan;     // FA path page scan: 0 exact, 1 hierarchical bisect
    int32_t  offload_attn_min_tokens; // floor on kept KV in TOKENS (0 = no floor)
    int32_t  offload_attn_head_reduce; // -1 unset (env/default), 0 union, 1 max
    bool     offload_attn_dense_listed; // true: the dense layers are the explicit set in offload_attn_dense_mask (resolved against n_layer)
    uint64_t offload_attn_dense_mask[4];
    int32_t  n_threads;      // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    std::vector<std::string> devices;
    // (devices_draft removed 2026-09-27: 0 readers, 0 writers. The draft model's devices travel as
    //  common_params_speculative.devices -> params_dft.devices at common/speculative.cpp:3092.)

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    float defrag_thold;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    // Mamba-2 gated group RMSNorm as one GGML_OP_GATED_GROUP_RMS_NORM node (mamba2, nemotron_h,
    // falcon_h1, granite hybrid). A layer whose device answers supports_op == false gets the
    // unfused chain instead. LLAMA_FUSED_GATED_NORM=0|1 overrides it (A/B).
    bool fused_gated_norm = true;
    bool fused_top_k_mask;   // use fused top-k sparse attention mask
    bool auto_ftkm;
    bool fused_kv_rows;      // unified cache: attend to the cells of each sequence only (ggml_flash_attn_ext_rows)
    bool auto_fkvr;
    bool no_perf;
    bool warmup;             // note: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;
    int  mla_attn;
    int  attn_max_batch;
    bool fused_moe_up_gate;
    bool grouped_expert_routing;
    bool fused_up_gate;
    bool fused_mmad;
    bool rope_cache;
    bool graph_reuse;
    bool prefetch_experts;
    bool k_cache_hadamard;
    bool v_cache_hadamard;
    int  kv_fresh_chunk = 0;          // see llama_context_params::kv_fresh_chunk (row S69-OV-PICK-FRESH-CHUNK-FP16-KV-OVER-QUANTISED-CACHE-2026-10-06-01)
    bool dsa_indexer_hadamard = true; // apply Walsh-Hadamard rotation to DSA indexer q/k (precision)
    bool dsa = false;                 // enable GLM DSA sparse attention (off by default; opt-in via --dsa)
    bool fused_idx_topk = false;      // enable the fused indexer topk op (set from llama_context_params, where it is on by default)
    bool swa_compress = false;
    bool dsv4_legacy_state = false;   // write DeepSeek-V4 state in the legacy full-slice layout (no MAGIC)
    int  dsa_top_k = -1;              // DSA top-k override (<0 => use the model's configured indexer_top_k)
    bool split_mode_graph_scheduling;
    // PRIOR-ART-NONE: pre-existing commented-out field carried verbatim from HEAD's
    // pristine llama-cparams.h (git show HEAD:src/llama-cparams.h) -- not a disablement
    // introduced by this merge resolution, left as-is.
    //bool split_mode_f16;
    bool scheduler_async;
    int  min_experts;
    float thresh_experts;
    size_t moe_cache_size = 0;
    float  moe_cache_cpu_share = 0.0f;
    int    moe_cache_policy = 0; // enum llama_moe_cache_policy
    bool mtp;
    int  worst_graph_tokens;
    int  dflash_query_capacity = 0; // internal DFlash query capacity override
    bool training = false;   // upstream: set by llama_opt_init()

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum ggml_type reduce_type;
    enum ggml_type graph_attn_precision;
    enum ggml_type idx_type_k = GGML_TYPE_F16;
    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;
    enum llama_mtp_op_type mtp_op_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    void * cuda_params;

    llama_context * ctx_other;

    // ── TriAttention KV scoring params (unified ik_llama build) ─────────────
    // Enable with --triattention <stats_path>. RESTORED 2026-08-20: the
    // implementation files survived the consolidation and compile into every
    // binary, but these fields and their argv parsing did not, so nothing could
    // switch the feature on -- declared, built, unreachable.
    std::string tri_stats_path;  // path to pre-computed calibration stats (empty = disabled)
    float       tri_budget;      // retention % (0.0-1.0), e.g. 0.75 = keep 75% of KV cache
    int         tri_window;      // recent N tokens always kept (default 512)
    int         tri_interval;    // score/prune every N tokens (default 128)
    int         tri_sink;        // first N tokens always kept as attention sinks (default 0)

    // ── CATS activation sparsity (S69-CATS-STAGE1-2026-09-28) ───────────────
    // arXiv:2404.08763 Sec 3.1: a = silu(gate(x)); a[|a| < t_l] = 0; out = down(a * up(x)).
    // t_l is per-layer, produced by the offline calibration tool (tools/cats-calibrate).
    // cats_sparsity == 0.0f is the off state and MUST take the exact pre-existing fused
    // graph path (bit-identity gate) -- see llm_build_ffn / llm_build_moe_ffn.
    float              cats_sparsity = 0.0f;   // target sparsity, informational (thresholds below drive the mask)
    std::vector<float> cats_thresholds;        // size n_layer; index il, 0.0f at an index = no mask that layer
    bool               cats_calibrate = false; // route the un-fused silu(gate) path WITHOUT masking, so
                                                // tools/cats-calibrate can read the raw "ffn_silu" tensor
                                                // via cb_eval on an unmodified model (chicken-and-egg fix:
                                                // the fused kernels never materialize this tensor at all).
    // ── MoE resident expert pool (S69-KT-EXPERT-POOL-2026-09-28) ─────────────
    // DISABLE-REVERSIBLE: moe_pool_slots == 0 is the off switch; new opt-in fields.
    // Port of llama.cpp PR #27861 (csantiago78) + discussion #28248 (memoriaru); design
    // note in common/common.h, call site in src/llama-build-context.cpp llm_build_lora_mm_id.
    int         moe_pool_slots      = 0;
    int         moe_pool_inserts    = 2;
    int         moe_pool_admit      = 3;
    bool        moe_pool_cpu_only   = false;
    std::string moe_stats_path;
    std::string moe_stats_mode      = "stat";
    std::string moe_placement_strategy = "uniform";
    std::string moe_placement_stats;
    bool        moe_dynamic_update  = false;
    int         moe_dynamic_threshold = 512;

    // FFD block-sparse decode (see llama_context_params::ffd)
    bool        ffd       = false;
    int32_t     ffd_block = 128;
    float       ffd_delta = 5.0f;

    // KIVI simulated low-bit KV (see llama_context_params::kivi_bits); 0 = off
    int32_t     kivi_bits     = 0;
    int32_t     kivi_group    = 32;
    int32_t     kivi_residual = 128;

    // XAttention prefill block selection (see llama_context_params::xattn_threshold); 0 = off
    float       xattn_threshold = 0.0f;
    int32_t     xattn_stride    = 8;
    int32_t     xattn_block     = 128;

    // KVCrush / AdaptiveR-KV retention for the SnapKV/H2O prompt prune (see llama_context_params::kv_diversity_mass); 0 = off
    float       kv_diversity_mass   = 0.0f;
    int32_t     kv_diversity_block  = 16;
    int32_t     kv_diversity_layers = 4;

    // SnapKV/H2O survivor re-base after the prune (see llama_context_params::snapkv_rebase); 0 = off
    int32_t     snapkv_rebase = 0;

    // ── TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28) ───────────────
    // arXiv:2408.14690 Sec 4.2 Definition 1: stp(x)_i = 0 if |x_i| <= tp else x_i, applied to the
    // INPUT of a linear projection (not its output, unlike CATS). teal_sparsity == 0.0f is the off
    // state and MUST take the exact pre-existing fused graph path (bit-identity gate) -- see
    // llm_build_ffn and the attention block builder in llama-build-context.cpp.
    // Three families of threshold, one per distinct input distribution actually wired this session
    // (uniform target sparsity across all matrices sharing that input -- no Algorithm 1 greedy
    // per-matrix reallocation, so q/k/v (and gate/up) sharing an input also share one threshold):
    float              teal_sparsity = 0.0f;        // target sparsity, informational
    std::vector<float> teal_thresholds_attn_in;      // size n_layer; input to Wq/Wk/Wv
    std::vector<float> teal_thresholds_ffn_in;       // size n_layer; input to Wgate/Wup
    std::vector<float> teal_thresholds_ffn_mid;      // size n_layer; input to Wdown (post silu*up)
    bool               teal_calibrate = false;       // route the un-fused FFN path WITHOUT masking, so
                                                      // the calibration tool can read "teal_ffn_mid-<il>"
                                                      // (never materialized by the fused kernels).
};
