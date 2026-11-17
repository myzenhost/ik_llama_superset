#include "llama-model.h"
#include "llama-context.h"
#include "llama-cparams.h"

// for ggml_backend_cuda_diffusion_sample (DiffusionGemma stage-1 device sampler)
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#  include "ggml-cuda.h"
#endif

#include <map>

// ADDED 2026-09-09 (DISPATCH_CREATE_MEMORY_2026_09_09) -- headers for the memory implementations
// upstream's create_memory constructs. Every one of these already existed in src/ EXCEPT
// llama-memory-hybrid-idx.h, which upstream has and this fork had never merged; it was fetched
// from upstream rather than reinvented. None of these .cpp files is in the build yet -- that is
// the all-or-nothing spine wiring, recorded separately -- but the DECLARATIONS are what this
// translation unit needs to compile.
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid-idx.h"

// ADDED 2026-09-09 (DISPATCH_BUILD_GRAPH_2026_09_09) -- the dispatcher's graph half.
// See the declarations in llama-model.h for why these two exist and what measured them safe.

ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    // An OLD-generation model: its graph is built by llm_build_context::llama_build_graph in
    // src/llama.cpp, not through this virtual. Returning nullptr is the answer llama-context.cpp
    // already expects -- it tests `if (!gf)` on the next line -- rather than a placeholder.
    GGML_UNUSED(params);
    return nullptr;
}

// PORTED VERBATIM from upstream ggml-org/llama.cpp master, src/llama-model.cpp
// (https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp, 153,165 chars,
// fetched raw and read in full), where it is llama_model::build_graph -- 716 chars, 19 lines. The
// body below is inserted from the extracted upstream text itself, not retyped, so it cannot drift.
// The ONLY change is the qualified name: llama_model_base:: here, because build_arch_graph is
// declared on llama_model_base in this fork and on llama_model upstream.
// 2026-11-05 upstream merge: upstream's body now also calls llm->build_logits() (MERGE_HEAD src/llama-model.cpp:3344-3365) --
// the model classes declare res->t_logits WITHOUT expanding it, and build_logits() is what adds the LM head to the graph. Without it
// every class-built graph ended at the last layer's attention: out_ids never entered the graph, got no buffer, and the first decode
// aborted at llm_graph_input_out_ids::set_input (ggml-backend.cpp GGML_ASSERT(buffer); ctest test-llama-archs / -parity /
// -deepstack-mapping-default, measured with GGML_SCHED_DEBUG=2: 65 nodes ending at layer 1 FLASH_ATTN).
ggml_cgraph * llama_model_base::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm);

    // add the LM head (if the context outputs logits)
    llm->build_logits();

    // add backend sampling layers (if any)
    llm->build_sampling();

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);

    llm->res->set_outputs(params);

    return llm->res->get_gf();
}

static const std::map<llm_arch, std::map<llm_tensor, std::string>> LLM_TENSOR_NAMES = {
    // MERGED 2026-09-09 (slice 47c) -- hy_v4, an architecture upstream has and
    // this fork never took. DERIVED, not copied: upstream keys tensor names FLAT
    // (llama-arch.cpp:427), this fork BY ARCH, so each entry pairs a tensor
    // hy-v4.cpp itself references with upstream's own string. LIMITATION: the set
    // is harvested from that file, so a tensor reached only through a helper
    // defined elsewhere would be absent -- and a short map throws at LOAD time.
    {
        LLM_ARCH_HY_V4,
        {
            { LLM_TENSOR_ATTN_GATE,                      "blk.%d.attn_gate" },
            { LLM_TENSOR_ATTN_KV_A_MQA,                  "blk.%d.attn_kv_a_mqa" },
            { LLM_TENSOR_ATTN_KV_A_NORM,                 "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_K_B,                       "blk.%d.attn_k_b" },
            { LLM_TENSOR_ATTN_NORM,                      "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_OUT,                       "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_A,                       "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_A_NORM,                  "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_Q_B,                       "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_SINKS,                     "blk.%d.attn_sinks" },
            { LLM_TENSOR_ATTN_V_B,                       "blk.%d.attn_v_b" },
            { LLM_TENSOR_FFN_DOWN,                       "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_DOWN_EXPS,                  "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,                 "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,                "blk.%d.exp_probs_b" },
            { LLM_TENSOR_FFN_GATE,                       "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_GATE_EXPS,                  "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_GATE_INP,                   "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,                 "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_NORM,                       "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,                         "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_UP_EXPS,                    "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_UP_SHEXP,                   "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_HC_ATTN_BASE,                   "blk.%d.hc_attn_base" },
            { LLM_TENSOR_HC_ATTN_FN,                     "blk.%d.hc_attn_fn" },
            { LLM_TENSOR_HC_ATTN_SCALE,                  "blk.%d.hc_attn_scale" },
            { LLM_TENSOR_HC_FFN_BASE,                    "blk.%d.hc_ffn_base" },
            { LLM_TENSOR_HC_FFN_FN,                      "blk.%d.hc_ffn_fn" },
            { LLM_TENSOR_HC_FFN_SCALE,                   "blk.%d.hc_ffn_scale" },
            { LLM_TENSOR_HC_HEAD_BASE,                   "output_hc_base" },
            { LLM_TENSOR_HC_HEAD_FN,                     "output_hc_fn" },
            { LLM_TENSOR_HC_HEAD_SCALE,                  "output_hc_scale" },
            { LLM_TENSOR_INDEXER_ATTN_K,                 "blk.%d.indexer.attn_k" },
            { LLM_TENSOR_INDEXER_ATTN_Q_B,               "blk.%d.indexer.attn_q_b" },
            { LLM_TENSOR_INDEXER_K_NORM,                 "blk.%d.indexer.k_norm" },
            { LLM_TENSOR_INDEXER_PROJ,                   "blk.%d.indexer.proj" },
            { LLM_TENSOR_OUTPUT,                         "output" },
            { LLM_TENSOR_OUTPUT_NORM,                    "output_norm" },
            { LLM_TENSOR_TOKEN_EMBD,                     "token_embd" },
        },
    },
    // MERGED 2026-09-09 (slice 47b) -- spark2_5, an architecture upstream has
    // and this fork never took. DERIVED, not copied: upstream keys tensor names
    // FLAT (llama-arch.cpp:427) while this fork keys them BY ARCH, so each entry
    // pairs a tensor spark2-5.cpp itself references with upstream's own string.
    // LIMITATION: the set is harvested from that file, so a tensor reached only
    // through a helper defined elsewhere would be absent -- and a short map
    // throws at LOAD time, not at compile time.
    {
        LLM_ARCH_SPARK2_5,
        {
            { LLM_TENSOR_ATTN_GATE,                      "blk.%d.attn_gate" },
            { LLM_TENSOR_ATTN_NORM,                      "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_OUT,                       "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,                       "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE,                       "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_NORM,                       "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,                         "blk.%d.ffn_up" },
            { LLM_TENSOR_OUTPUT,                         "output" },
            { LLM_TENSOR_OUTPUT_NORM,                    "output_norm" },
            { LLM_TENSOR_TOKEN_EMBD,                     "token_embd" },
        },
    },
    {
        LLM_ARCH_LLAMA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
        },
    },
{
        LLM_ARCH_K2_HORIZON,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_ATTN_GATE,       "blk.%d.attn_gate" },
            { LLM_TENSOR_ATTN_V_GATE,     "blk.%d.attn_v_gate" },
            { LLM_TENSOR_ATTN_V_EXPS,     "blk.%d.attn_v_exps" },
            { LLM_TENSOR_FFN_EXP_PROBS_B, "blk.%d.exp_probs_b" },
            { LLM_TENSOR_FFN_GATE_SHEXP,  "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,  "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,    "blk.%d.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_MUSE_GLIMMER,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            //{ LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.%d.post_attention_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_GATE,       "blk.%d.attn_gate" },
            //{ LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_DECI,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
        },
    },
    {
        LLM_ARCH_LLAMA4,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_SHEXP,  "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,  "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,    "blk.%d.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_BAICHUAN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_FALCON,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_NORM_2,     "blk.%d.attn_norm_2" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_GROK,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.%d.layer_output_norm" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
        },
    },
    {
        LLM_ARCH_GPT2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_GPTJ,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_GPTNEOX,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_MPT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output"},
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_ACT,         "blk.%d.ffn.act" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm"},
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm"},
        },
    },
    {
        LLM_ARCH_STARCODER,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_REFACT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_BERT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_TOKEN_TYPES,     "token_types" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.%d.layer_output_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_CLS,             "cls" },
            { LLM_TENSOR_CLS_OUT,         "cls.output" },
        },
    },
    {
        LLM_ARCH_NOMIC_BERT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_TOKEN_TYPES,     "token_types" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.%d.layer_output_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_CLS,             "cls" },
            { LLM_TENSOR_CLS_OUT,         "cls.output" },
        },
    },
    {
        LLM_ARCH_JINA_BERT_V2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_TOKEN_TYPES,     "token_types" },
            { LLM_TENSOR_ATTN_NORM_2,     "blk.%d.attn_norm_2" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.%d.layer_output_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_CLS,             "cls" },
            { LLM_TENSOR_CLS_OUT,         "cls.output" },
        },
    },
    {
        LLM_ARCH_BLOOM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_STABLELM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
        },
    },
    {
        LLM_ARCH_QWEN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN2MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_QWEN2VL,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN3,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_CLS_OUT,         "cls.output" }, // rerank head, as upstream models/qwen3.cpp
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_ATTN_Q_NORM_IN,    "blk.%d.attn_q_norm_in" },
            { LLM_TENSOR_ATTN_K_NORM_IN,    "blk.%d.attn_k_norm_in" },
            { LLM_TENSOR_ATTN_V_NORM_IN,    "blk.%d.attn_v_norm_in" },
            { LLM_TENSOR_ATTN_OUT_NORM_IN,  "blk.%d.attn_output_norm_in" },
            { LLM_TENSOR_FFN_GATE_NORM_IN,  "blk.%d.ffn_gate_norm_in" },
            { LLM_TENSOR_FFN_UP_NORM_IN,    "blk.%d.ffn_up_norm_in" },
            { LLM_TENSOR_FFN_DOWN_NORM_IN,  "blk.%d.ffn_down_norm_in" },
            { LLM_TENSOR_FFN_TTT_CONV,    "blk.%d.ffn_ttt_conv" }, // In-Place TTT (S69-FORK-INPLACE-TTT-2026-10-26-01)
            { LLM_TENSOR_FFN_TTT_PROJ,    "blk.%d.ffn_ttt_proj" },
        },
    },
    {
        LLM_ARCH_QWEN3MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
        },
    },
    {
        LLM_ARCH_MELLUM,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_QWEN4EXP,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "per_layer_token_embd" },
            { LLM_TENSOR_HC_HEAD_NORM,         "output_hc_norm" },
            { LLM_TENSOR_HC_HEAD_DOWN,         "output_hc_down" },
            { LLM_TENSOR_HC_HEAD_UP,           "output_hc_up" },
            { LLM_TENSOR_NEXTN_EH_PROJ,        "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_ENORM,          "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,          "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
            { LLM_TENSOR_NEXTN_HC_HEAD_NORM,   "blk.%d.nextn.hc_head_norm" },
            { LLM_TENSOR_NEXTN_HC_HEAD_DOWN,   "blk.%d.nextn.hc_head_down" },
            { LLM_TENSOR_NEXTN_HC_HEAD_UP,     "blk.%d.nextn.hc_head_up" },
            { LLM_TENSOR_ATTN_Q,               "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,               "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,               "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,             "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,          "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,          "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_QKV,             "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_GATE,            "blk.%d.attn_gate" },
            { LLM_TENSOR_INDEXER_Q_PROJ,       "blk.%d.indexer.q_proj" },
            { LLM_TENSOR_INDEXER_K_PROJ,       "blk.%d.indexer.k_proj" },
            { LLM_TENSOR_INDEXER_Q_NORM,       "blk.%d.indexer.q_norm" },
            { LLM_TENSOR_INDEXER_K_NORM,       "blk.%d.indexer.k_norm" },
            { LLM_TENSOR_SSM_CONV1D,           "blk.%d.ssm_conv1d" },
            { LLM_TENSOR_SSM_DT,               "blk.%d.ssm_dt" },
            { LLM_TENSOR_SSM_A_NOSCAN,         "blk.%d.ssm_a" },
            { LLM_TENSOR_SSM_ALPHA,            "blk.%d.ssm_alpha" },
            { LLM_TENSOR_SSM_BETA,             "blk.%d.ssm_beta" },
            { LLM_TENSOR_SSM_NORM,             "blk.%d.ssm_norm" },
            { LLM_TENSOR_SSM_OUT,              "blk.%d.ssm_out" },
            { LLM_TENSOR_HC_ATTN_NORM,         "blk.%d.hc_attn_norm" },
            { LLM_TENSOR_HC_ATTN_DOWN,         "blk.%d.hc_attn_down" },
            { LLM_TENSOR_HC_ATTN_UP,           "blk.%d.hc_attn_up" },
            { LLM_TENSOR_HC_ATTN_INJECT,       "blk.%d.hc_attn_inject" },
            { LLM_TENSOR_HC_FFN_NORM,          "blk.%d.hc_ffn_norm" },
            { LLM_TENSOR_HC_FFN_DOWN,          "blk.%d.hc_ffn_down" },
            { LLM_TENSOR_HC_FFN_UP,            "blk.%d.hc_ffn_up" },
            { LLM_TENSOR_HC_FFN_INJECT,        "blk.%d.hc_ffn_inject" },
            { LLM_TENSOR_PLE_KEY,              "blk.%d.ple_key" },
            { LLM_TENSOR_PLE_VALUE,            "blk.%d.ple_value" },
            { LLM_TENSOR_PLE_NORM_KEY,         "blk.%d.ple_norm_key" },
            { LLM_TENSOR_PLE_NORM_QUERY,       "blk.%d.ple_norm_query" },
            { LLM_TENSOR_PLE_NORM_CONV,        "blk.%d.ple_norm_conv" },
            { LLM_TENSOR_PLE_CONV1D,           "blk.%d.ple_conv1d" },
            { LLM_TENSOR_FFN_GATE_INP,         "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,        "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,        "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,          "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,     "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP,   "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,       "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,       "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,         "blk.%d.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_QWEN3NEXT,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_POST_NORM,     "blk.%d.post_attention_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_QKV,           "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_GATE,          "blk.%d.attn_gate" },
            { LLM_TENSOR_SSM_CONV1D,         "blk.%d.ssm_conv1d" },
            { LLM_TENSOR_SSM_DT,             "blk.%d.ssm_dt" },
            { LLM_TENSOR_SSM_A_NOSCAN,       "blk.%d.ssm_a" },
            { LLM_TENSOR_SSM_BETA_ALPHA,     "blk.%d.ssm_ba" },
            { LLM_TENSOR_SSM_IN,             "blk.%d.ssm_in" },
            { LLM_TENSOR_SSM_NORM,           "blk.%d.ssm_norm" },
            { LLM_TENSOR_SSM_OUT,            "blk.%d.ssm_out" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_QWEN35MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_POST_NORM,     "blk.%d.post_attention_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_QKV,           "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_GATE,          "blk.%d.attn_gate" },
            { LLM_TENSOR_SSM_CONV1D,         "blk.%d.ssm_conv1d" },
            { LLM_TENSOR_SSM_DT,             "blk.%d.ssm_dt" },
            { LLM_TENSOR_SSM_A_NOSCAN,       "blk.%d.ssm_a" },
            { LLM_TENSOR_SSM_BETA,           "blk.%d.ssm_beta" },
            { LLM_TENSOR_SSM_ALPHA,          "blk.%d.ssm_alpha" },
            //{ LLM_TENSOR_SSM_IN,             "blk.%d.ssm_in" },
            { LLM_TENSOR_SSM_NORM,           "blk.%d.ssm_norm" },
            { LLM_TENSOR_SSM_OUT,            "blk.%d.ssm_out" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_NEXTN_EH_PROJ,      "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_ENORM,        "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,        "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
        },
    },
    {
        LLM_ARCH_QWEN35,
        {
            { LLM_TENSOR_TOKEN_EMBD,             "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,            "output_norm" },
            { LLM_TENSOR_OUTPUT,                 "output" },
            { LLM_TENSOR_ATTN_NORM,              "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_POST_NORM,         "blk.%d.post_attention_norm" },
            { LLM_TENSOR_ATTN_Q,                 "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,            "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,                 "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,            "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,                 "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,               "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_QKV,               "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_GATE,              "blk.%d.attn_gate" },
            { LLM_TENSOR_SSM_CONV1D,             "blk.%d.ssm_conv1d" },
            { LLM_TENSOR_SSM_DT,                 "blk.%d.ssm_dt" },
            { LLM_TENSOR_SSM_A_NOSCAN,           "blk.%d.ssm_a" },
            { LLM_TENSOR_SSM_BETA,               "blk.%d.ssm_beta" },
            { LLM_TENSOR_SSM_ALPHA,              "blk.%d.ssm_alpha" },
            { LLM_TENSOR_SSM_NORM,               "blk.%d.ssm_norm" },
            { LLM_TENSOR_SSM_OUT,                "blk.%d.ssm_out" },
            { LLM_TENSOR_FFN_GATE,               "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,               "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,                 "blk.%d.ffn_up" },
            { LLM_TENSOR_NEXTN_EH_PROJ,          "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_ENORM,            "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,            "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
        },
    },
    {
        LLM_ARCH_QWEN3VL,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_CLS_OUT,         "cls.output" }, // rerank head, as upstream models/qwen3vl.cpp
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN3VLMOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
        },
    },
    {
        LLM_ARCH_PHI2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_PHI3,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ROPE_FACTORS_LONG,  "rope_factors_long" },
            { LLM_TENSOR_ROPE_FACTORS_SHORT, "rope_factors_short" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,           "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_PLAMO,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_CODESHELL,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_ORION,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_INTERNLM2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_MINICPM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            // MiniCPM4 longrope: conversion/minicpm.py generate_extra_tensors writes these two (global, no block index).
            // Without the entries tn() returns "__missing__" and the GGUF's tensors are never requested.
            { LLM_TENSOR_ROPE_FACTORS_LONG,  "rope_factors_long" },
            { LLM_TENSOR_ROPE_FACTORS_SHORT, "rope_factors_short" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
        },
    },
    {
        LLM_ARCH_GEMMA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_GEMMA2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.%d.post_ffw_norm" },
        },
    },
    {
        LLM_ARCH_GEMMA3,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_ATTN_Q_NORM_IN,    "blk.%d.attn_q_norm_in" },
            { LLM_TENSOR_ATTN_K_NORM_IN,    "blk.%d.attn_k_norm_in" },
            { LLM_TENSOR_ATTN_V_NORM_IN,    "blk.%d.attn_v_norm_in" },
            { LLM_TENSOR_ATTN_OUT_NORM_IN,  "blk.%d.attn_output_norm_in" },
            { LLM_TENSOR_FFN_GATE_NORM_IN,  "blk.%d.ffn_gate_norm_in" },
            { LLM_TENSOR_FFN_UP_NORM_IN,    "blk.%d.ffn_up_norm_in" },
            { LLM_TENSOR_FFN_DOWN_NORM_IN,  "blk.%d.ffn_down_norm_in" },
        },
    },
    {
        // gemma-embedding (embeddinggemma): gemma3's block tensors, no lm_head, plus the two
        // SentenceTransformers dense projections applied after pooling.
        LLM_ARCH_GEMMA_EMBEDDING,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_DENSE_2_OUT,     "dense_2" },
            { LLM_TENSOR_DENSE_3_OUT,     "dense_3" },
        },
    },
    {
        LLM_ARCH_GEMMA4,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,          "output_norm" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_ROPE_FREQS,           "rope_freqs" },
            { LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "per_layer_token_embd" },
            { LLM_TENSOR_PER_LAYER_MODEL_PROJ, "per_layer_model_proj" },
            { LLM_TENSOR_PER_LAYER_PROJ_NORM,  "per_layer_proj_norm" },
            { LLM_TENSOR_ATTN_NORM,            "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,               "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,          "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,               "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,          "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,               "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,             "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,       "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,             "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,             "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,             "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,               "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,     "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,        "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_GATE_INP,         "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_POST_NORM,        "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_FFN_POST_NORM_1,      "blk.%d.post_ffw_norm_1" },
            { LLM_TENSOR_FFN_POST_NORM_2,      "blk.%d.post_ffw_norm_2" },
            { LLM_TENSOR_FFN_PRE_NORM_2,       "blk.%d.pre_ffw_norm_2" },
            { LLM_TENSOR_LAYER_OUT_SCALE,      "blk.%d.layer_output_scale" },
            { LLM_TENSOR_PER_LAYER_INP_GATE,   "blk.%d.inp_gate" },
            { LLM_TENSOR_PER_LAYER_PROJ,       "blk.%d.proj" },
            { LLM_TENSOR_PER_LAYER_POST_NORM,  "blk.%d.post_norm" },
        },
    },
    {
        // DiffusionGemma reuses the whole gemma4 backbone naming; the only additions are the
        // encoder per-layer scalar and the model-level self-conditioning gated MLP.
        LLM_ARCH_DIFFUSION_GEMMA,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,          "output_norm" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_ROPE_FREQS,           "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,            "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,               "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,          "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,               "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,          "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,               "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,             "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,       "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,             "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,             "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,             "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,               "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,     "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_EXPS,        "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,          "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,        "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_GATE_INP,         "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_POST_NORM,        "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_FFN_POST_NORM_1,      "blk.%d.post_ffw_norm_1" },
            { LLM_TENSOR_FFN_POST_NORM_2,      "blk.%d.post_ffw_norm_2" },
            { LLM_TENSOR_FFN_PRE_NORM_2,       "blk.%d.pre_ffw_norm_2" },
            { LLM_TENSOR_LAYER_OUT_SCALE,      "blk.%d.layer_output_scale" },
            { LLM_TENSOR_ENC_LAYER_OUT_SCALE,  "blk.%d.enc_layer_output_scale" },
            { LLM_TENSOR_SC_PRE_NORM,          "self_cond_pre_norm" },
            { LLM_TENSOR_SC_GATE,              "self_cond_gate" },
            { LLM_TENSOR_SC_UP,                "self_cond_up" },
            { LLM_TENSOR_SC_DOWN,              "self_cond_down" },
        },
    },
    {
        LLM_ARCH_GEMMA4_MTP,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,          "output_norm" },
            { LLM_TENSOR_ATTN_NORM,            "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,               "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,          "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_OUT,             "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,       "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,             "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,             "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,             "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,               "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_POST_NORM,        "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_LAYER_OUT_SCALE,      "blk.%d.layer_output_scale" },
            { LLM_TENSOR_MTP_PRE_PROJ,         "mtp_pre_proj" },
            { LLM_TENSOR_MTP_POST_PROJ,        "mtp_post_proj" },
            { LLM_TENSOR_MTP_TOKEN_ORDERING,   "mtp_token_ordering" },
            { LLM_TENSOR_MTP_CENTROIDS,        "mtp_centroids" },
        },
    },
    {
        LLM_ARCH_DFLASH_DRAFT,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,          "output_norm" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_ATTN_NORM,            "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,               "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,          "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,               "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,          "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,               "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_GATE,            "blk.%d.attn_gate" },
            { LLM_TENSOR_ATTN_SINKS,           "blk.%d.attn_sinks" },
            { LLM_TENSOR_ATTN_OUT,             "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,       "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_GATE,             "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,             "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,               "blk.%d.ffn_up" },
            { LLM_TENSOR_DFLASH_FC,            "dflash_fc" },
            { LLM_TENSOR_DFLASH_HIDDEN_NORM,   "dflash_hidden_norm" },
            { LLM_TENSOR_DFLASH_AUX_HIDDEN_NORM, "dflash_aux_hidden_norm.%d" },
            { LLM_TENSOR_DSPARK_MARKOV_W1,     "markov_w1" },
            { LLM_TENSOR_DSPARK_MARKOV_W2,     "markov_w2" },
            { LLM_TENSOR_DSPARK_CONF_PROJ,     "conf_proj" },
        },
    },
    {
        LLM_ARCH_DFLASH,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,          "output_norm" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_ATTN_NORM,            "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,               "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,          "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,               "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,          "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,               "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,             "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_SINKS,           "blk.%d.attn_sinks" },
            { LLM_TENSOR_ATTN_Q_A_NORM,        "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_KV_A_NORM,       "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_Q_A,             "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_B,             "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_LATENT,       "blk.%d.attn_kv" },
            { LLM_TENSOR_ATTN_OUT_A,           "blk.%d.attn_output_a" },
            { LLM_TENSOR_ATTN_OUT_B,           "blk.%d.attn_output_b" },
            { LLM_TENSOR_FFN_NORM,             "blk.%d.ffn_norm" },
            { LLM_TENSOR_ATTN_POST_NORM,       "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,         "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,        "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,        "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,          "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_SHEXP,       "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,       "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,         "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,      "blk.%d.exp_probs_b" },
            { LLM_TENSOR_HC_HEAD_BASE,         "output_hc_base" },
            { LLM_TENSOR_HC_HEAD_FN,           "output_hc_fn" },
            { LLM_TENSOR_HC_HEAD_SCALE,        "output_hc_scale" },
            { LLM_TENSOR_HC_ATTN_BASE,         "blk.%d.hc_attn_base" },
            { LLM_TENSOR_HC_ATTN_FN,           "blk.%d.hc_attn_fn" },
            { LLM_TENSOR_HC_ATTN_SCALE,        "blk.%d.hc_attn_scale" },
            { LLM_TENSOR_HC_FFN_BASE,          "blk.%d.hc_ffn_base" },
            { LLM_TENSOR_HC_FFN_FN,            "blk.%d.hc_ffn_fn" },
            { LLM_TENSOR_HC_FFN_SCALE,         "blk.%d.hc_ffn_scale" },
            { LLM_TENSOR_FFN_GATE,             "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,             "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,               "blk.%d.ffn_up" },
            { LLM_TENSOR_DFLASH_FC,            "fc" },
            { LLM_TENSOR_DFLASH_HIDDEN_NORM,   "enc.output_norm" },
            { LLM_TENSOR_DSPARK_MARKOV_W1,     "markov_w1" },
            { LLM_TENSOR_DSPARK_MARKOV_W2,     "markov_w2" },
            { LLM_TENSOR_DSPARK_CONF_PROJ,     "conf_proj" },
        },
    },
    {
        LLM_ARCH_DFLASH2,
        {
            { LLM_TENSOR_TOKEN_EMBD,             "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,             "output_norm" },
            { LLM_TENSOR_OUTPUT,                  "output" },
            { LLM_TENSOR_ATTN_NORM,               "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,                  "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,             "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,                  "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,             "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,                  "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,                "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,                "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,                "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,                "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,                  "blk.%d.ffn_up" },
            { LLM_TENSOR_DFLASH_FC,               "fc" },
            { LLM_TENSOR_DFLASH_HIDDEN_NORM,      "enc.output_norm" },
            { LLM_TENSOR_DFLASH_ATTN_CONV_BASE,   "blk.%d.attn_conv_base" },
            { LLM_TENSOR_DFLASH_ATTN_CONV_PROJ,   "blk.%d.attn_conv_proj" },
            { LLM_TENSOR_DFLASH_FFN_CONV_BASE,    "blk.%d.ffn_conv_base" },
            { LLM_TENSOR_DFLASH_FFN_CONV_PROJ,    "blk.%d.ffn_conv_proj" },
            { LLM_TENSOR_DFLASH_SELECTOR_PREV,    "selector_predecessor" },
            { LLM_TENSOR_DFLASH_SELECTOR_NEXT,    "selector_successor" },
            { LLM_TENSOR_DFLASH_SELECTOR_HIDDEN,  "selector_hidden" },
        },
    },
    {
        LLM_ARCH_GEMMA4_ASSISTANT,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,          "output_norm" },
            { LLM_TENSOR_ATTN_NORM,            "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,               "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,          "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_OUT,             "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,       "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,             "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,             "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,             "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,               "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_POST_NORM,        "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_LAYER_OUT_SCALE,      "blk.%d.layer_output_scale" },
            { LLM_TENSOR_MTP_PRE_PROJ,         "mtp_pre_proj" },
            { LLM_TENSOR_MTP_POST_PROJ,        "mtp_post_proj" },
            { LLM_TENSOR_MTP_TOKEN_ORDERING,   "mtp_token_ordering" },
            { LLM_TENSOR_MTP_CENTROIDS,        "mtp_centroids" },
            { LLM_TENSOR_ROPE_FREQS,           "rope_freqs" },
        },
    },
    {
        LLM_ARCH_STARCODER2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_MAMBA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_SSM_IN,          "blk.%d.ssm_in" },
            { LLM_TENSOR_SSM_CONV1D,      "blk.%d.ssm_conv1d" },
            { LLM_TENSOR_SSM_X,           "blk.%d.ssm_x" },
            { LLM_TENSOR_SSM_DT,          "blk.%d.ssm_dt" },
            { LLM_TENSOR_SSM_A,           "blk.%d.ssm_a" },
            { LLM_TENSOR_SSM_D,           "blk.%d.ssm_d" },
            { LLM_TENSOR_SSM_OUT,         "blk.%d.ssm_out" },
        },
    },
    {
        LLM_ARCH_GRANITE_HYBRID,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_SSM_IN,          "blk.%d.ssm_in" },
            { LLM_TENSOR_SSM_CONV1D,      "blk.%d.ssm_conv1d" },
            { LLM_TENSOR_SSM_DT,          "blk.%d.ssm_dt" },
            { LLM_TENSOR_SSM_A,           "blk.%d.ssm_a" },
            { LLM_TENSOR_SSM_D,           "blk.%d.ssm_d" },
            { LLM_TENSOR_SSM_NORM,        "blk.%d.ssm_norm" },
            { LLM_TENSOR_SSM_OUT,         "blk.%d.ssm_out" },
        },
    },
    {
        LLM_ARCH_XVERSE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_COMMAND_R,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
        },
    },
    {
        LLM_ARCH_DBRX,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
        },
    },
    {
        LLM_ARCH_OLMO,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_OPENELM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_ARCTIC,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_NORM_EXPS,   "blk.%d.ffn_norm_exps" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
        },
    },
    {
        LLM_ARCH_DEEPSEEK2,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q_A_NORM,      "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_KV_A_NORM,     "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_A,           "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_B,           "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_A_MQA,      "blk.%d.attn_kv_a_mqa" },
            { LLM_TENSOR_ATTN_KQ_A_MQA,      "blk.%d.attn_kq_a_mqa" },
            { LLM_TENSOR_ATTN_KV_B,          "blk.%d.attn_kv_b" },
            { LLM_TENSOR_ATTN_K_B,           "blk.%d.attn_k_b" },
            { LLM_TENSOR_ATTN_V_B,           "blk.%d.attn_v_b" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_DEEPSEEK4,
        {
            { LLM_TENSOR_TOKEN_EMBD,             "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,            "output_norm" },
            { LLM_TENSOR_OUTPUT,                 "output" },
            { LLM_TENSOR_ATTN_NORM,              "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_SINKS,             "blk.%d.attn_sinks" },
            { LLM_TENSOR_ATTN_Q_A_NORM,          "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_KV_A_NORM,         "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_Q,                 "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_A,               "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_B,               "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_LATENT,         "blk.%d.attn_kv" },
            { LLM_TENSOR_ATTN_OUT_A,             "blk.%d.attn_output_a" },
            { LLM_TENSOR_ATTN_OUT_B,             "blk.%d.attn_output_b" },
            { LLM_TENSOR_ATTN_COMP_KV,           "blk.%d.attn_compressor_kv" },
            { LLM_TENSOR_ATTN_COMP_GATE,         "blk.%d.attn_compressor_gate" },
            { LLM_TENSOR_ATTN_COMP_APE,          "blk.%d.attn_compressor_ape" },
            { LLM_TENSOR_ATTN_COMP_NORM,         "blk.%d.attn_compressor_norm" },
            { LLM_TENSOR_ATTN_OUT,               "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,               "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,               "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,                 "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,               "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE_INP,           "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,          "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,          "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,            "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,       "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP,     "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,         "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,         "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,           "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,        "blk.%d.exp_probs_b" },
            { LLM_TENSOR_FFN_GATE_TID2EID,       "blk.%d.ffn_gate_tid2eid" },
            { LLM_TENSOR_INDEXER_K_NORM,         "blk.%d.indexer.k_norm" },
            { LLM_TENSOR_INDEXER_PROJ,           "blk.%d.indexer.proj" },
            { LLM_TENSOR_INDEXER_ATTN_K,         "blk.%d.indexer.attn_k" },
            { LLM_TENSOR_INDEXER_ATTN_Q_B,       "blk.%d.indexer.attn_q_b" },
            { LLM_TENSOR_INDEXER_COMP_KV,        "blk.%d.indexer_compressor_kv" },
            { LLM_TENSOR_INDEXER_COMP_GATE,      "blk.%d.indexer_compressor_gate" },
            { LLM_TENSOR_INDEXER_COMP_APE,       "blk.%d.indexer_compressor_ape" },
            { LLM_TENSOR_INDEXER_COMP_NORM,      "blk.%d.indexer_compressor_norm" },
            { LLM_TENSOR_HC_HEAD_BASE,           "output_hc_base" },
            { LLM_TENSOR_HC_HEAD_FN,             "output_hc_fn" },
            { LLM_TENSOR_HC_HEAD_SCALE,          "output_hc_scale" },
            { LLM_TENSOR_HC_ATTN_BASE,           "blk.%d.hc_attn_base" },
            { LLM_TENSOR_HC_ATTN_FN,             "blk.%d.hc_attn_fn" },
            { LLM_TENSOR_HC_ATTN_SCALE,          "blk.%d.hc_attn_scale" },
            { LLM_TENSOR_HC_FFN_BASE,            "blk.%d.hc_ffn_base" },
            { LLM_TENSOR_HC_FFN_FN,              "blk.%d.hc_ffn_fn" },
            { LLM_TENSOR_HC_FFN_SCALE,           "blk.%d.hc_ffn_scale" },
            { LLM_TENSOR_NEXTN_EH_PROJ,          "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_EMBED_TOKENS,     "blk.%d.nextn.embed_tokens" },
            { LLM_TENSOR_NEXTN_ENORM,            "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,            "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "blk.%d.nextn.shared_head_head" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
        },
    },
    {
        LLM_ARCH_DEEPSEEK41,
        {
            { LLM_TENSOR_TOKEN_EMBD,             "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,            "output_norm" },
            { LLM_TENSOR_OUTPUT,                 "output" },
            { LLM_TENSOR_ATTN_NORM,              "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_SINKS,             "blk.%d.attn_sinks" },
            { LLM_TENSOR_ATTN_Q_A_NORM,          "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_KV_A_NORM,         "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_Q,                 "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_A,               "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_B,               "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_LATENT,         "blk.%d.attn_kv" },
            { LLM_TENSOR_ATTN_OUT_A,             "blk.%d.attn_output_a" },
            { LLM_TENSOR_ATTN_OUT_B,             "blk.%d.attn_output_b" },
            { LLM_TENSOR_ATTN_COMP_KV,           "blk.%d.attn_compressor_kv" },
            { LLM_TENSOR_ATTN_COMP_GATE,         "blk.%d.attn_compressor_gate" },
            { LLM_TENSOR_ATTN_COMP_APE,          "blk.%d.attn_compressor_ape" },
            { LLM_TENSOR_ATTN_COMP_NORM,         "blk.%d.attn_compressor_norm" },
            { LLM_TENSOR_ATTN_OUT,               "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,               "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,               "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,                 "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,               "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE_INP,           "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,          "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,          "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,            "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,       "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP,     "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,         "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,         "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,           "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,        "blk.%d.exp_probs_b" },
            { LLM_TENSOR_FFN_GATE_TID2EID,       "blk.%d.ffn_gate_tid2eid" },
            { LLM_TENSOR_INDEXER_K_NORM,         "blk.%d.indexer.k_norm" },
            { LLM_TENSOR_INDEXER_PROJ,           "blk.%d.indexer.proj" },
            { LLM_TENSOR_INDEXER_ATTN_K,         "blk.%d.indexer.attn_k" },
            { LLM_TENSOR_INDEXER_ATTN_Q_B,       "blk.%d.indexer.attn_q_b" },
            { LLM_TENSOR_INDEXER_COMP_KV,        "blk.%d.indexer_compressor_kv" },
            { LLM_TENSOR_INDEXER_COMP_GATE,      "blk.%d.indexer_compressor_gate" },
            { LLM_TENSOR_INDEXER_COMP_APE,       "blk.%d.indexer_compressor_ape" },
            { LLM_TENSOR_INDEXER_COMP_NORM,      "blk.%d.indexer_compressor_norm" },
            { LLM_TENSOR_HC_HEAD_BASE,           "output_hc_base" },
            { LLM_TENSOR_HC_HEAD_FN,             "output_hc_fn" },
            { LLM_TENSOR_HC_HEAD_SCALE,          "output_hc_scale" },
            { LLM_TENSOR_HC_ATTN_BASE,           "blk.%d.hc_attn_base" },
            { LLM_TENSOR_HC_ATTN_FN,             "blk.%d.hc_attn_fn" },
            { LLM_TENSOR_HC_ATTN_SCALE,          "blk.%d.hc_attn_scale" },
            { LLM_TENSOR_HC_FFN_BASE,            "blk.%d.hc_ffn_base" },
            { LLM_TENSOR_HC_FFN_FN,              "blk.%d.hc_ffn_fn" },
            { LLM_TENSOR_HC_FFN_SCALE,           "blk.%d.hc_ffn_scale" },
            { LLM_TENSOR_NEXTN_EH_PROJ,          "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_EMBED_TOKENS,     "blk.%d.nextn.embed_tokens" },
            { LLM_TENSOR_NEXTN_ENORM,            "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,            "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "blk.%d.nextn.shared_head_head" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
            { LLM_TENSOR_ENGRAM_EMBD,            "blk.%d.engram_embd" },
            { LLM_TENSOR_ENGRAM_K,               "blk.%d.engram_k" },
            { LLM_TENSOR_ENGRAM_Q,               "blk.%d.engram_q" },
            { LLM_TENSOR_ENGRAM_WKV,             "blk.%d.engram_wkv" },
        },
    },
    {
        LLM_ARCH_MISTRAL4,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q_A_NORM,      "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_KV_A_NORM,     "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_A,           "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_B,           "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_A_MQA,      "blk.%d.attn_kv_a_mqa" },
            { LLM_TENSOR_ATTN_KQ_A_MQA,      "blk.%d.attn_kq_a_mqa" },
            { LLM_TENSOR_ATTN_KV_B,          "blk.%d.attn_kv_b" },
            { LLM_TENSOR_ATTN_K_B,           "blk.%d.attn_k_b" },
            { LLM_TENSOR_ATTN_V_B,           "blk.%d.attn_v_b" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_CHATGLM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_GLM4,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.%d.post_ffw_norm" },
        },
    },
    {
        LLM_ARCH_GLM4_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_POST_NORM,     "blk.%d.post_attention_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
            // NextN/MTP tensors - preserved but unused (in final layer, dynamic layer number)
            { LLM_TENSOR_NEXTN_EH_PROJ,      "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_EMBED_TOKENS, "blk.%d.nextn.embed_tokens" },
            { LLM_TENSOR_NEXTN_ENORM,        "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,        "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "blk.%d.nextn.shared_head_head" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
        },
    },
    {
        LLM_ARCH_BITNET,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_SUB_NORM,      "blk.%d.attn_sub_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_SUB_NORM,       "blk.%d.ffn_sub_norm" },
        },
    },
    {
        LLM_ARCH_BITNET_25,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_ATTN_SUB_NORM,   "blk.%d.attn_sub_norm" },
            { LLM_TENSOR_FFN_SUB_NORM,    "blk.%d.ffn_sub_norm" },
        },
    },
    {
        LLM_ARCH_BITNET_B158,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_ATTN_SUB_NORM,   "blk.%d.attn_sub_norm" },
            { LLM_TENSOR_FFN_SUB_NORM,    "blk.%d.ffn_sub_norm" },
        },
    },
    {
        LLM_ARCH_T5,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_DEC_OUTPUT_NORM,      "dec.output_norm" },
            { LLM_TENSOR_DEC_ATTN_NORM,        "dec.blk.%d.attn_norm" },
            { LLM_TENSOR_DEC_ATTN_Q,           "dec.blk.%d.attn_q" },
            { LLM_TENSOR_DEC_ATTN_K,           "dec.blk.%d.attn_k" },
            { LLM_TENSOR_DEC_ATTN_V,           "dec.blk.%d.attn_v" },
            { LLM_TENSOR_DEC_ATTN_OUT,         "dec.blk.%d.attn_o" },
            { LLM_TENSOR_DEC_ATTN_REL_B,       "dec.blk.%d.attn_rel_b" },
            { LLM_TENSOR_DEC_CROSS_ATTN_NORM,  "dec.blk.%d.cross_attn_norm" },
            { LLM_TENSOR_DEC_CROSS_ATTN_Q,     "dec.blk.%d.cross_attn_q" },
            { LLM_TENSOR_DEC_CROSS_ATTN_K,     "dec.blk.%d.cross_attn_k" },
            { LLM_TENSOR_DEC_CROSS_ATTN_V,     "dec.blk.%d.cross_attn_v" },
            { LLM_TENSOR_DEC_CROSS_ATTN_OUT,   "dec.blk.%d.cross_attn_o" },
            { LLM_TENSOR_DEC_CROSS_ATTN_REL_B, "dec.blk.%d.cross_attn_rel_b" },
            { LLM_TENSOR_DEC_FFN_NORM,         "dec.blk.%d.ffn_norm" },
            { LLM_TENSOR_DEC_FFN_GATE,         "dec.blk.%d.ffn_gate" },
            { LLM_TENSOR_DEC_FFN_DOWN,         "dec.blk.%d.ffn_down" },
            { LLM_TENSOR_DEC_FFN_UP,           "dec.blk.%d.ffn_up" },
            { LLM_TENSOR_ENC_OUTPUT_NORM,      "enc.output_norm" },
            { LLM_TENSOR_ENC_ATTN_NORM,        "enc.blk.%d.attn_norm" },
            { LLM_TENSOR_ENC_ATTN_Q,           "enc.blk.%d.attn_q" },
            { LLM_TENSOR_ENC_ATTN_K,           "enc.blk.%d.attn_k" },
            { LLM_TENSOR_ENC_ATTN_V,           "enc.blk.%d.attn_v" },
            { LLM_TENSOR_ENC_ATTN_OUT,         "enc.blk.%d.attn_o" },
            { LLM_TENSOR_ENC_ATTN_REL_B,       "enc.blk.%d.attn_rel_b" },
            { LLM_TENSOR_ENC_FFN_NORM,         "enc.blk.%d.ffn_norm" },
            { LLM_TENSOR_ENC_FFN_GATE,         "enc.blk.%d.ffn_gate" },
            { LLM_TENSOR_ENC_FFN_DOWN,         "enc.blk.%d.ffn_down" },
            { LLM_TENSOR_ENC_FFN_UP,           "enc.blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_T5ENCODER,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_ENC_OUTPUT_NORM,      "enc.output_norm" },
            { LLM_TENSOR_ENC_ATTN_NORM,        "enc.blk.%d.attn_norm" },
            { LLM_TENSOR_ENC_ATTN_Q,           "enc.blk.%d.attn_q" },
            { LLM_TENSOR_ENC_ATTN_K,           "enc.blk.%d.attn_k" },
            { LLM_TENSOR_ENC_ATTN_V,           "enc.blk.%d.attn_v" },
            { LLM_TENSOR_ENC_ATTN_OUT,         "enc.blk.%d.attn_o" },
            { LLM_TENSOR_ENC_ATTN_REL_B,       "enc.blk.%d.attn_rel_b" },
            { LLM_TENSOR_ENC_FFN_NORM,         "enc.blk.%d.ffn_norm" },
            { LLM_TENSOR_ENC_FFN_GATE,         "enc.blk.%d.ffn_gate" },
            { LLM_TENSOR_ENC_FFN_DOWN,         "enc.blk.%d.ffn_down" },
            { LLM_TENSOR_ENC_FFN_UP,           "enc.blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_JAIS,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_GRANITE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_GRANITE_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
        },
    },
    {
        LLM_ARCH_COHERE2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_COHERE2_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
        },
    },
    {
        LLM_ARCH_DOTS1,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
        }
    },
    {
        LLM_ARCH_ERNIE4_5,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_ERNIE4_5_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_HUNYUAN_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_SHEXP,  "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,  "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,    "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
        },
    },
    {
        LLM_ARCH_OPENAI_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_POST_NORM,     "blk.%d.post_attention_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_SINKS,         "blk.%d.attn_sinks" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
        },
    },
    {
        LLM_ARCH_BAILINGMOE2,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_QKV,           "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_NEXTN_EH_PROJ,      "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_EMBED_TOKENS, "blk.%d.nextn.embed_tokens" },
            { LLM_TENSOR_NEXTN_ENORM,        "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,        "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "blk.%d.nextn.shared_head_head" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
            { LLM_TENSOR_LAYER_OUT_NORM,     "blk.%d.layer_output_norm" },
        },
    },
    {
        LLM_ARCH_BAILINGMOE3,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_GATE,          "blk.%d.attn_gate" },
            { LLM_TENSOR_ATTN_Q_A,           "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_A_NORM,      "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_Q_B,           "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_A_MQA,      "blk.%d.attn_kv_a_mqa" },
            { LLM_TENSOR_ATTN_KV_A_NORM,     "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_K_B,           "blk.%d.attn_k_b" },
            { LLM_TENSOR_ATTN_V_B,           "blk.%d.attn_v_b" },
            { LLM_TENSOR_SSM_CONV1D_Q,       "blk.%d.ssm_conv1d_q" },
            { LLM_TENSOR_SSM_CONV1D_K,       "blk.%d.ssm_conv1d_k" },
            { LLM_TENSOR_SSM_CONV1D_V,       "blk.%d.ssm_conv1d_v" },
            { LLM_TENSOR_SSM_F_A,            "blk.%d.ssm_f_a" },
            { LLM_TENSOR_SSM_G_A,            "blk.%d.ssm_g_a" },
            { LLM_TENSOR_SSM_F,              "blk.%d.ssm_f" },
            { LLM_TENSOR_SSM_G,              "blk.%d.ssm_g" },
            { LLM_TENSOR_SSM_A,              "blk.%d.ssm_a" },
            { LLM_TENSOR_SSM_DT,             "blk.%d.ssm_dt" },
            { LLM_TENSOR_SSM_BETA,           "blk.%d.ssm_beta" },
            { LLM_TENSOR_SSM_NORM,           "blk.%d.ssm_norm" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_NEXTN_EH_PROJ,      "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_ENORM,        "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,        "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_LAYER_OUT_NORM,     "blk.%d.layer_output_norm" },
        },
    },
    {
        LLM_ARCH_KIMI_K3,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd"               },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm"              },
            { LLM_TENSOR_OUTPUT,             "output"                   },
            { LLM_TENSOR_OUTPUT_RES_SCORE,   "output_res_score"         },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm"         },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q"            },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k"            },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v"            },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output"       },
            { LLM_TENSOR_ATTN_GATE,          "blk.%d.attn_gate"         },
            { LLM_TENSOR_ATTN_Q_A,           "blk.%d.attn_q_a"          },
            { LLM_TENSOR_ATTN_Q_A_NORM,      "blk.%d.attn_q_a_norm"     },
            { LLM_TENSOR_ATTN_Q_B,           "blk.%d.attn_q_b"          },
            { LLM_TENSOR_ATTN_KV_A_MQA,      "blk.%d.attn_kv_a_mqa"     },
            { LLM_TENSOR_ATTN_KV_A_NORM,     "blk.%d.attn_kv_a_norm"    },
            { LLM_TENSOR_ATTN_K_B,           "blk.%d.attn_k_b"          },
            { LLM_TENSOR_ATTN_V_B,           "blk.%d.attn_v_b"          },
            { LLM_TENSOR_ATTN_RES_SCORE,     "blk.%d.attn_res_score"    },
            { LLM_TENSOR_SSM_CONV1D_Q,       "blk.%d.ssm_conv1d_q"      },
            { LLM_TENSOR_SSM_CONV1D_K,       "blk.%d.ssm_conv1d_k"      },
            { LLM_TENSOR_SSM_CONV1D_V,       "blk.%d.ssm_conv1d_v"      },
            { LLM_TENSOR_SSM_F_A,            "blk.%d.ssm_f_a"           },
            { LLM_TENSOR_SSM_F_B,            "blk.%d.ssm_f_b"           },
            { LLM_TENSOR_SSM_G,              "blk.%d.ssm_g"             },
            { LLM_TENSOR_SSM_A,              "blk.%d.ssm_a"             },
            { LLM_TENSOR_SSM_DT,             "blk.%d.ssm_dt"            },
            { LLM_TENSOR_SSM_BETA,           "blk.%d.ssm_beta"          },
            { LLM_TENSOR_SSM_NORM,           "blk.%d.ssm_norm"          },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm"          },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate"          },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down"          },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up"            },
            { LLM_TENSOR_FFN_RES_SCORE,      "blk.%d.ffn_res_score"     },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp"      },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b"       },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps"     },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps"     },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps"       },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp"    },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp"    },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp"      },
            { LLM_TENSOR_FFN_ROUTED_DOWN,    "blk.%d.ffn_routed_down"   },
            { LLM_TENSOR_FFN_ROUTED_UP,      "blk.%d.ffn_routed_up"     },
            { LLM_TENSOR_FFN_ROUTED_NORM,    "blk.%d.ffn_routed_norm"   },
        },
    },
    {
        LLM_ARCH_LFM2,
        {
            { LLM_TENSOR_TOKEN_EMBD,             "token_embd"                   },
            { LLM_TENSOR_TOKEN_EMBD_NORM,        "token_embd_norm"              },
            { LLM_TENSOR_OUTPUT,                 "output"                       },
            { LLM_TENSOR_ATTN_NORM,              "blk.%d.attn_norm"             },
            { LLM_TENSOR_ATTN_Q,                 "blk.%d.attn_q"                },
            { LLM_TENSOR_ATTN_K,                 "blk.%d.attn_k"                },
            { LLM_TENSOR_ATTN_V,                 "blk.%d.attn_v"                },
            { LLM_TENSOR_ATTN_OUT,               "blk.%d.attn_output"           },
            { LLM_TENSOR_ATTN_Q_NORM,            "blk.%d.attn_q_norm"           },
            { LLM_TENSOR_ATTN_K_NORM,            "blk.%d.attn_k_norm"           },
            { LLM_TENSOR_FFN_NORM,               "blk.%d.ffn_norm"              },
            { LLM_TENSOR_FFN_GATE,               "blk.%d.ffn_gate"              },
            { LLM_TENSOR_FFN_DOWN,               "blk.%d.ffn_down"              },
            { LLM_TENSOR_FFN_UP,                 "blk.%d.ffn_up"                },
            { LLM_TENSOR_SHORTCONV_CONV,         "blk.%d.shortconv.conv"        },
            { LLM_TENSOR_SHORTCONV_INPROJ,       "blk.%d.shortconv.in_proj"     },
            { LLM_TENSOR_SHORTCONV_OUTPROJ,      "blk.%d.shortconv.out_proj"    },
        },
    },
    {
        LLM_ARCH_LFM2MOE,   // ik_llama #2530: LFM2 + routed experts
        {
            { LLM_TENSOR_TOKEN_EMBD,             "token_embd"                   },
            { LLM_TENSOR_TOKEN_EMBD_NORM,        "token_embd_norm"              },
            { LLM_TENSOR_OUTPUT,                 "output"                       },
            { LLM_TENSOR_ATTN_NORM,              "blk.%d.attn_norm"             },
            { LLM_TENSOR_ATTN_Q,                 "blk.%d.attn_q"                },
            { LLM_TENSOR_ATTN_K,                 "blk.%d.attn_k"                },
            { LLM_TENSOR_ATTN_V,                 "blk.%d.attn_v"                },
            { LLM_TENSOR_ATTN_OUT,               "blk.%d.attn_output"           },
            { LLM_TENSOR_ATTN_Q_NORM,            "blk.%d.attn_q_norm"           },
            { LLM_TENSOR_ATTN_K_NORM,            "blk.%d.attn_k_norm"           },
            { LLM_TENSOR_FFN_NORM,               "blk.%d.ffn_norm"              },
            { LLM_TENSOR_FFN_GATE,               "blk.%d.ffn_gate"              },
            { LLM_TENSOR_FFN_DOWN,               "blk.%d.ffn_down"              },
            { LLM_TENSOR_FFN_UP,                 "blk.%d.ffn_up"                },
            { LLM_TENSOR_FFN_GATE_INP,           "blk.%d.ffn_gate_inp"          },
            { LLM_TENSOR_FFN_GATE_EXPS,          "blk.%d.ffn_gate_exps"         },
            { LLM_TENSOR_FFN_DOWN_EXPS,          "blk.%d.ffn_down_exps"         },
            { LLM_TENSOR_FFN_UP_EXPS,            "blk.%d.ffn_up_exps"           },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,       "blk.%d.ffn_gate_up_exps"      },
            { LLM_TENSOR_FFN_EXP_PROBS_B,        "blk.%d.exp_probs_b"           },
            { LLM_TENSOR_SHORTCONV_CONV,         "blk.%d.shortconv.conv"        },
            { LLM_TENSOR_SHORTCONV_INPROJ,       "blk.%d.shortconv.in_proj"     },
            { LLM_TENSOR_SHORTCONV_OUTPROJ,      "blk.%d.shortconv.out_proj"    },
        },
    },
    {
        // S69 P4: Inkling. SOURCE-READ: src__llama-arch.cpp:676-688 (names) and gguf-py constants.py:4834-4865 (tensor set) of the
        // saved unslothai/llama.cpp inkling-25731-b10775 (MIT). The shared-expert banks are 3D ({n_embd, n_ff_exp, n_shexp}) but
        // carry the ordinary *_shexp names.
        LLM_ARCH_INKLING,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM,    "token_embd_norm" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_R,             "blk.%d.attn_r" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_REL_PROJ,      "blk.%d.attn_rel_proj" },
            { LLM_TENSOR_SHORTCONV_K,        "blk.%d.shortconv_k" },
            { LLM_TENSOR_SHORTCONV_V,        "blk.%d.shortconv_v" },
            { LLM_TENSOR_SHORTCONV_ATTN,     "blk.%d.shortconv_attn" },
            { LLM_TENSOR_SHORTCONV_MLP,      "blk.%d.shortconv_mlp" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GSCALE,         "blk.%d.ffn_gscale" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_MINIMAX_M2,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_MINIMAX_M3,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_INDEXER_Q_PROJ,     "blk.%d.indexer.q_proj" },
            { LLM_TENSOR_INDEXER_K_PROJ,     "blk.%d.indexer.k_proj" },
            { LLM_TENSOR_INDEXER_Q_NORM,     "blk.%d.indexer.q_norm" },
            { LLM_TENSOR_INDEXER_K_NORM,     "blk.%d.indexer.k_norm" },
        },
    },
    {
        LLM_ARCH_SMOLLM3,
        {
            { LLM_TENSOR_TOKEN_EMBD,     "token_embd"            },
            { LLM_TENSOR_OUTPUT_NORM,    "output_norm"           },
            { LLM_TENSOR_OUTPUT,         "output"                },
            { LLM_TENSOR_ATTN_NORM,      "blk.%d.attn_norm"      },
            { LLM_TENSOR_ATTN_Q,         "blk.%d.attn_q"         },
            { LLM_TENSOR_ATTN_K,         "blk.%d.attn_k"         },
            { LLM_TENSOR_ATTN_V,         "blk.%d.attn_v"         },
            { LLM_TENSOR_ATTN_OUT,       "blk.%d.attn_output"    },
            { LLM_TENSOR_FFN_NORM,       "blk.%d.ffn_norm"       },
            { LLM_TENSOR_FFN_GATE,       "blk.%d.ffn_gate"       },
            { LLM_TENSOR_FFN_DOWN,       "blk.%d.ffn_down"       },
            { LLM_TENSOR_FFN_UP,         "blk.%d.ffn_up"         },
        },
    },
    {
        // HyperCLOVA X (S69-EXLLAMAV3-PICK-HYPERCLOVAX-2026-10-20-01): the llama tensor set plus the Peri-LN post norms.
        // The GGUF names are the Gemma-2 ones (post_attention_norm / post_ffw_norm) so gguf-py needs no new tensor kinds.
        LLM_ARCH_HYPERCLOVAX,
        {
            { LLM_TENSOR_TOKEN_EMBD,     "token_embd"            },
            { LLM_TENSOR_OUTPUT_NORM,    "output_norm"           },
            { LLM_TENSOR_OUTPUT,         "output"                },
            { LLM_TENSOR_ATTN_NORM,      "blk.%d.attn_norm"      },
            { LLM_TENSOR_ATTN_Q,         "blk.%d.attn_q"         },
            { LLM_TENSOR_ATTN_K,         "blk.%d.attn_k"         },
            { LLM_TENSOR_ATTN_V,         "blk.%d.attn_v"         },
            { LLM_TENSOR_ATTN_OUT,       "blk.%d.attn_output"    },
            { LLM_TENSOR_ATTN_POST_NORM, "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,       "blk.%d.ffn_norm"       },
            { LLM_TENSOR_FFN_GATE,       "blk.%d.ffn_gate"       },
            { LLM_TENSOR_FFN_DOWN,       "blk.%d.ffn_down"       },
            { LLM_TENSOR_FFN_UP,         "blk.%d.ffn_up"         },
            { LLM_TENSOR_FFN_POST_NORM,  "blk.%d.post_ffw_norm"  },
        },
    },
    {
        LLM_ARCH_MISTRAL3,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
        },
    },
    {
        LLM_ARCH_MIMO2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_SINKS,      "blk.%d.attn_sinks" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_NEXTN_EH_PROJ,      "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_ENORM,        "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,        "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_LAYER_OUT_NORM,     "blk.%d.layer_output_norm" },
            { LLM_TENSOR_FFN_EXP_PROBS_B, "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_SEED_OSS,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_STEP35,
        {
            {   LLM_TENSOR_TOKEN_EMBD,        "token_embd" },
            {   LLM_TENSOR_OUTPUT_NORM,       "output_norm" },
            {   LLM_TENSOR_OUTPUT,            "output" },
            {   LLM_TENSOR_ROPE_FREQS,        "rope_freqs" },
            {   LLM_TENSOR_ROPE_FACTORS_LONG, "rope_factors_long" },
            {   LLM_TENSOR_ROPE_FACTORS_SHORT,"rope_factors_short" },
            {   LLM_TENSOR_ATTN_NORM,         "blk.%d.attn_norm" },
            {   LLM_TENSOR_ATTN_Q,            "blk.%d.attn_q" },
            {   LLM_TENSOR_ATTN_Q_NORM,       "blk.%d.attn_q_norm" },
            {   LLM_TENSOR_ATTN_K,            "blk.%d.attn_k" },
            {   LLM_TENSOR_ATTN_K_NORM,       "blk.%d.attn_k_norm" },
            {   LLM_TENSOR_ATTN_V,            "blk.%d.attn_v" },
            {   LLM_TENSOR_ATTN_GATE,         "blk.%d.attn_gate" },
            {   LLM_TENSOR_ATTN_OUT,          "blk.%d.attn_output" },
            {   LLM_TENSOR_FFN_NORM,          "blk.%d.ffn_norm" },
            {   LLM_TENSOR_FFN_GATE,          "blk.%d.ffn_gate" },
            {   LLM_TENSOR_FFN_DOWN,          "blk.%d.ffn_down" },
            {   LLM_TENSOR_FFN_UP,            "blk.%d.ffn_up" },
            {   LLM_TENSOR_FFN_GATE_INP,      "blk.%d.ffn_gate_inp" },
            {   LLM_TENSOR_FFN_GATE_EXPS,     "blk.%d.ffn_gate_exps" },
            {   LLM_TENSOR_FFN_DOWN_EXPS,     "blk.%d.ffn_down_exps" },
            {   LLM_TENSOR_FFN_UP_EXPS,       "blk.%d.ffn_up_exps" },
            {   LLM_TENSOR_FFN_GATE_UP_EXPS,  "blk.%d.ffn_gate_up_exps" },
            {   LLM_TENSOR_FFN_GATE_SHEXP,    "blk.%d.ffn_gate_shexp" },
            {   LLM_TENSOR_FFN_DOWN_SHEXP,    "blk.%d.ffn_down_shexp" },
            {   LLM_TENSOR_FFN_UP_SHEXP,      "blk.%d.ffn_up_shexp" },
            {   LLM_TENSOR_FFN_EXP_PROBS_B,   "blk.%d.exp_probs_b" },
            {   LLM_TENSOR_NEXTN_EH_PROJ,          "blk.%d.nextn.eh_proj" },
            {   LLM_TENSOR_NEXTN_EMBED_TOKENS,     "blk.%d.nextn.embed_tokens" },
            {   LLM_TENSOR_NEXTN_ENORM,             "blk.%d.nextn.enorm" },
            {   LLM_TENSOR_NEXTN_HNORM,             "blk.%d.nextn.hnorm" },
            {   LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "blk.%d.nextn.shared_head_head" },
            {   LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
        },
    },
    {
        LLM_ARCH_LAGUNA,
        {
            {   LLM_TENSOR_TOKEN_EMBD,        "token_embd" },
            {   LLM_TENSOR_OUTPUT_NORM,       "output_norm" },
            {   LLM_TENSOR_OUTPUT,            "output" },
            {   LLM_TENSOR_ROPE_FREQS,        "rope_freqs" },
            {   LLM_TENSOR_ROPE_FACTORS_LONG, "rope_factors_long" },
            {   LLM_TENSOR_ROPE_FACTORS_SHORT,"rope_factors_short" },
            {   LLM_TENSOR_ATTN_NORM,         "blk.%d.attn_norm" },
            {   LLM_TENSOR_ATTN_Q,            "blk.%d.attn_q" },
            {   LLM_TENSOR_ATTN_Q_NORM,       "blk.%d.attn_q_norm" },
            {   LLM_TENSOR_ATTN_K,            "blk.%d.attn_k" },
            {   LLM_TENSOR_ATTN_K_NORM,       "blk.%d.attn_k_norm" },
            {   LLM_TENSOR_ATTN_V,            "blk.%d.attn_v" },
            {   LLM_TENSOR_ATTN_GATE,         "blk.%d.attn_gate" },
            {   LLM_TENSOR_ATTN_OUT,          "blk.%d.attn_output" },
            {   LLM_TENSOR_FFN_NORM,          "blk.%d.ffn_norm" },
            {   LLM_TENSOR_FFN_GATE,          "blk.%d.ffn_gate" },
            {   LLM_TENSOR_FFN_DOWN,          "blk.%d.ffn_down" },
            {   LLM_TENSOR_FFN_UP,            "blk.%d.ffn_up" },
            {   LLM_TENSOR_FFN_GATE_INP,      "blk.%d.ffn_gate_inp" },
            {   LLM_TENSOR_FFN_GATE_EXPS,     "blk.%d.ffn_gate_exps" },
            {   LLM_TENSOR_FFN_DOWN_EXPS,     "blk.%d.ffn_down_exps" },
            {   LLM_TENSOR_FFN_UP_EXPS,       "blk.%d.ffn_up_exps" },
            {   LLM_TENSOR_FFN_GATE_UP_EXPS,  "blk.%d.ffn_gate_up_exps" },
            {   LLM_TENSOR_FFN_GATE_SHEXP,    "blk.%d.ffn_gate_shexp" },
            {   LLM_TENSOR_FFN_DOWN_SHEXP,    "blk.%d.ffn_down_shexp" },
            {   LLM_TENSOR_FFN_UP_SHEXP,      "blk.%d.ffn_up_shexp" },
            {   LLM_TENSOR_FFN_EXP_PROBS_B,   "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_GLM_DSA,
        {
            { LLM_TENSOR_TOKEN_EMBD,             "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,            "output_norm" },
            { LLM_TENSOR_OUTPUT,                 "output" },
            { LLM_TENSOR_ATTN_NORM,              "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q_A_NORM,          "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_KV_A_NORM,         "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_Q,                 "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_A,               "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_B,               "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_A_MQA,          "blk.%d.attn_kv_a_mqa" },
            { LLM_TENSOR_ATTN_KQ_A_MQA,          "blk.%d.attn_kq_a_mqa" },
            { LLM_TENSOR_ATTN_KV_B,              "blk.%d.attn_kv_b" },
            { LLM_TENSOR_ATTN_K_B,               "blk.%d.attn_k_b" },
            { LLM_TENSOR_ATTN_V_B,               "blk.%d.attn_v_b" },
            { LLM_TENSOR_ATTN_OUT,               "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,               "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,               "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,                 "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,               "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE_INP,           "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,          "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,          "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,            "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,       "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP,     "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,         "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,         "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,           "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,        "blk.%d.exp_probs_b" },
            { LLM_TENSOR_INDEXER_K_NORM,         "blk.%d.indexer.k_norm" },
            { LLM_TENSOR_INDEXER_PROJ,           "blk.%d.indexer.proj" },
            { LLM_TENSOR_INDEXER_ATTN_K,         "blk.%d.indexer.attn_k" },
            { LLM_TENSOR_INDEXER_ATTN_Q_B,       "blk.%d.indexer.attn_q_b" },
            { LLM_TENSOR_NEXTN_EH_PROJ,          "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_EMBED_TOKENS,     "blk.%d.nextn.embed_tokens" },
            { LLM_TENSOR_NEXTN_ENORM,            "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,            "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "blk.%d.nextn.shared_head_head" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },

        },
    },
    {
        LLM_ARCH_GLM5NEXT,
        {
            { LLM_TENSOR_TOKEN_EMBD,             "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,            "output_norm" },
            { LLM_TENSOR_OUTPUT,                 "output" },
            { LLM_TENSOR_ATTN_NORM,              "blk.%d.attn_norm" },
            { LLM_TENSOR_FFN_NORM,               "blk.%d.ffn_norm" },
            // KDA (kimi-k3 low-rank) layers: KDA q/k/v project to ssm_d_inner, out proj back to n_embd
            { LLM_TENSOR_ATTN_Q,                 "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,                 "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,                 "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,               "blk.%d.attn_output" },
            { LLM_TENSOR_SSM_CONV1D_Q,           "blk.%d.ssm_conv1d_q" },
            { LLM_TENSOR_SSM_CONV1D_K,           "blk.%d.ssm_conv1d_k" },
            { LLM_TENSOR_SSM_CONV1D_V,           "blk.%d.ssm_conv1d_v" },
            { LLM_TENSOR_SSM_F_A,                "blk.%d.ssm_f_a" },
            { LLM_TENSOR_SSM_F_B,                "blk.%d.ssm_f_b" },
            { LLM_TENSOR_SSM_G_A,                "blk.%d.ssm_g_a" },
            { LLM_TENSOR_SSM_G_B,                "blk.%d.ssm_g_b" },
            { LLM_TENSOR_SSM_BETA,               "blk.%d.ssm_beta" },
            { LLM_TENSOR_SSM_A,                  "blk.%d.ssm_a" },
            { LLM_TENSOR_SSM_DT,                 "blk.%d.ssm_dt" },
            { LLM_TENSOR_SSM_NORM,               "blk.%d.ssm_norm" },
            // MLA layers (nope-only absorbed)
            { LLM_TENSOR_ATTN_Q_A,               "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_A_NORM,          "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_Q_B,               "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_A_MQA,          "blk.%d.attn_kv_a_mqa" },
            { LLM_TENSOR_ATTN_KV_A_NORM,         "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_K_B,               "blk.%d.attn_k_b" },
            { LLM_TENSOR_ATTN_V_B,               "blk.%d.attn_v_b" },
            // DSA lightning indexer + k-pool compressor
            { LLM_TENSOR_INDEXER_K_NORM,         "blk.%d.indexer.k_norm" },
            { LLM_TENSOR_INDEXER_PROJ,           "blk.%d.indexer.proj" },
            { LLM_TENSOR_INDEXER_ATTN_K,         "blk.%d.indexer.attn_k" },
            { LLM_TENSOR_INDEXER_ATTN_Q_B,       "blk.%d.indexer.attn_q_b" },
            { LLM_TENSOR_INDEXER_COMP_GATE,      "blk.%d.indexer_compressor_gate" },
            { LLM_TENSOR_INDEXER_COMP_APE,       "blk.%d.indexer_compressor_ape" },
            // hyper-connections (mHC)
            { LLM_TENSOR_HC_ATTN_FN,             "blk.%d.hc_attn_fn" },
            { LLM_TENSOR_HC_ATTN_BASE,           "blk.%d.hc_attn_base" },
            { LLM_TENSOR_HC_ATTN_SCALE,          "blk.%d.hc_attn_scale" },
            { LLM_TENSOR_HC_FFN_FN,              "blk.%d.hc_ffn_fn" },
            { LLM_TENSOR_HC_FFN_BASE,            "blk.%d.hc_ffn_base" },
            { LLM_TENSOR_HC_FFN_SCALE,           "blk.%d.hc_ffn_scale" },
            // dense-lead FFN + MoE (sigmoid, shared expert)
            { LLM_TENSOR_FFN_GATE,               "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,               "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,                 "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,           "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,        "blk.%d.exp_probs_b" },
            { LLM_TENSOR_FFN_GATE_EXPS,         "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,         "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,           "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_SHEXP,         "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,         "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,           "blk.%d.ffn_up_shexp" },
            // NextN / MTP tail (embed_tokens + shared_head_head are tied to trunk in this model)
            { LLM_TENSOR_NEXTN_EH_PROJ,          "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_ENORM,            "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,            "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
        },
    },
    {
        LLM_ARCH_OPENPANGU,
        {
            { LLM_TENSOR_TOKEN_EMBD,             "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,            "output_norm" },
            { LLM_TENSOR_OUTPUT,                 "output" },
            { LLM_TENSOR_ATTN_NORM,              "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q_A,               "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_A_NORM,          "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_Q_B,               "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_A_MQA,          "blk.%d.attn_kv_a_mqa" },
            { LLM_TENSOR_ATTN_KV_A_NORM,         "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_KV_B,              "blk.%d.attn_kv_b" },
            { LLM_TENSOR_ATTN_K_B,               "blk.%d.attn_k_b" },
            { LLM_TENSOR_ATTN_V_B,               "blk.%d.attn_v_b" },
            { LLM_TENSOR_ATTN_OUT,               "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,         "blk.%d.post_attention_norm" },
            // DSA lightning indexer (loaded; unused in the dense-fallback graph)
            { LLM_TENSOR_INDEXER_K_NORM,         "blk.%d.attn_indexer_k_norm" },
            { LLM_TENSOR_INDEXER_PROJ,           "blk.%d.attn_indexer_weights_proj" },
            { LLM_TENSOR_INDEXER_ATTN_K,         "blk.%d.attn_indexer_k" },
            { LLM_TENSOR_INDEXER_ATTN_Q_B,       "blk.%d.attn_indexer_q_b" },
            // MoME causal convs + param sink
            { LLM_TENSOR_ATTN_QA_CONV,           "blk.%d.attn_qa_conv" },
            { LLM_TENSOR_ATTN_KV_CONV,           "blk.%d.attn_compresskv_conv" },
            { LLM_TENSOR_ATTN_O_CONV,            "blk.%d.attn_o_conv" },
            { LLM_TENSOR_ATTN_PARAM_SINK_KV,     "blk.%d.attn_param_sink_kv" },
            { LLM_TENSOR_ATTN_PARAM_SINK_K_PE,   "blk.%d.attn_param_sink_k_pe" },
            // FFN (dense-lead + MoE, sandwich norms)
            { LLM_TENSOR_FFN_NORM,               "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_POST_NORM,          "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_FFN_GATE,               "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,                 "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,               "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE_INP,           "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,          "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,          "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,            "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_SHEXP,         "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,         "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,           "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,        "blk.%d.exp_probs_b" },
            // mHC / Hyper-Connections
            { LLM_TENSOR_MHC_ATTN_PHI,           "blk.%d.attn_mhc_phi" },
            { LLM_TENSOR_MHC_ATTN_ALPHA,         "blk.%d.attn_mhc_alpha" },
            { LLM_TENSOR_MHC_ATTN_BETA,          "blk.%d.attn_mhc_beta" },
            { LLM_TENSOR_MHC_ATTN_GAMMA,         "blk.%d.attn_mhc_gamma" },
            { LLM_TENSOR_MHC_MLP_PHI,            "blk.%d.mlp_mhc_phi" },
            { LLM_TENSOR_MHC_MLP_ALPHA,          "blk.%d.mlp_mhc_alpha" },
            { LLM_TENSOR_MHC_MLP_BETA,           "blk.%d.mlp_mhc_beta" },
            { LLM_TENSOR_MHC_MLP_GAMMA,          "blk.%d.mlp_mhc_gamma" },
            { LLM_TENSOR_MHC_MERGE_PHI,          "merge_mhc_phi" },
            { LLM_TENSOR_MHC_MERGE_ALPHA,        "merge_mhc_alpha" },
            { LLM_TENSOR_MHC_MERGE_BETA,         "merge_mhc_beta" },
            { LLM_TENSOR_MHC_MERGE_GAMMA,        "merge_mhc_gamma" },
            { LLM_TENSOR_BLOCK_POST_NORM,        "blk.%d.block_post_norm" },
            // NextN / MTP tail
            { LLM_TENSOR_NEXTN_EH_PROJ,          "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_EMBED_TOKENS,     "blk.%d.nextn.embed_tokens" },
            { LLM_TENSOR_NEXTN_ENORM,            "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,            "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "blk.%d.nextn.shared_head_head" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
        },
    },
    {
        // S69 #5/#32 — EAGLE3 draft head (PR #18039 port). Single decoder layer (blk.0);
        // fc fuses 3 target hidden states; output projects to draft vocab; d2t maps draft->target.
        LLM_ARCH_EAGLE3,
        {
            { LLM_TENSOR_FC,              "fc" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_D2T,             "d2t" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_NORM_2,     "blk.%d.attn_norm_2" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        // S69 #MODERNBERT: pre-norm GeGLU encoder. attn_qkv packed; ffn_up = GeGLU Wi (gate+up); no biases.
        LLM_ARCH_MODERNBERT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
        },
    },
    {
        LLM_ARCH_UNKNOWN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        },
    },
};

// Architectures built as src/models classes have no per-arch table above; their tensor names are
// arch-independent, so they share this one (generated from upstream's global table, see the .inc header).
static const std::map<llm_tensor, std::string> LLM_TENSOR_NAMES_GENERIC = {
#include "llama-tensor-names-generic.inc"
};

static const std::map<llm_tensor, std::string> & llm_tensor_names_for(llm_arch arch) {
    auto it = LLM_TENSOR_NAMES.find(arch);
    return it != LLM_TENSOR_NAMES.end() ? it->second : LLM_TENSOR_NAMES_GENERIC;
}

static const std::string * llm_tensor_name_lookup(llm_arch arch, llm_tensor tensor, bool generic_first = false) {
    // generic_first (LLM_TN::generic_first): a class-built model whose arch keeps a legacy table with different names
    // (LLM_ARCH_DFLASH) takes upstream's generic names first; anything they lack still resolves through the arch table.
    if (generic_first) {
        auto g = LLM_TENSOR_NAMES_GENERIC.find(tensor);
        if (g != LLM_TENSOR_NAMES_GENERIC.end()) {
            return &g->second;
        }
    }
    const auto & map = llm_tensor_names_for(arch);
    auto it = map.find(tensor);
    if (it != map.end()) {
        return &it->second;
    }
    // An arch built as a src/models class may still have a per-arch table (spark2_5, hy_v4) that predates the class and lacks tensors the class creates;
    // those fall back to the generic names. Legacy archs keep "absent from the table == absent from the file" (granite: see llama-model-loader.cpp).
    // LLM_ARCH_DFLASH also has a class (llama_model_dflash, LiLiCorr drafts) that creates tensors its legacy table lacks
    // (fc, enc.output_norm, lilicorr.*): those fall back too. The legacy DFlash loader only creates names its table has.
    if (&map != &LLM_TENSOR_NAMES_GENERIC && (llama_arch_is_class_dispatched(arch) || arch == LLM_ARCH_DFLASH)) {
        auto g = LLM_TENSOR_NAMES_GENERIC.find(tensor);
        if (g != LLM_TENSOR_NAMES_GENERIC.end()) {
            return &g->second;
        }
    }
    return nullptr;
}

llm_tn_name LLM_TN::operator()(llm_tensor tensor) const {
    const std::string * n = llm_tensor_name_lookup(arch, tensor, generic_first);
    return n ? *n : "__missing__";
}

llm_tn_name LLM_TN::operator()(llm_tensor tensor, const std::string & suffix) const {
    const std::string * n = llm_tensor_name_lookup(arch, tensor, generic_first);
    return n ? *n + "." + suffix : "__missing__";
}

llm_tn_name LLM_TN::operator()(llm_tensor tensor, int bid) const {
    const std::string * n = llm_tensor_name_lookup(arch, tensor, generic_first);
    return n ? ::format(n->c_str(), bid) : "__missing__";
}

llm_tn_name LLM_TN::operator()(llm_tensor tensor, const std::string & suffix, int bid) const {
    const std::string * n = llm_tensor_name_lookup(arch, tensor, generic_first);
    return n ? ::format(n->c_str(), bid) + "." + suffix : "__missing__";
}

llm_tn_name LLM_TN::operator()(llm_tensor tensor, const std::string & suffix, int bid, int xid) const {
    const std::string * n = llm_tensor_name_lookup(arch, tensor, generic_first);
    return n ? ::format(n->c_str(), bid, xid) + "." + suffix : "__missing__";
}

void llama_model::set_tensor_overrides(const llama_model_params& params) {
    if (params.n_gpu_layers > 0) {
        tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
    }
}

std::string llama_model_ftype_name(llama_ftype ftype) {
    if (ftype & LLAMA_FTYPE_GUESSED) {
        return llama_model_ftype_name((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) + " (guessed)";
    }

    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:         return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:      return "F16";
        case LLAMA_FTYPE_MOSTLY_BF16:     return "BF16";
        case LLAMA_FTYPE_MOSTLY_TQ1_0:    return "TQ1_0 - 1.69 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_TQ2_0:    return "TQ2_0 - 2.06 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_BF16_R16: return "BF16_R16";
        case LLAMA_FTYPE_MOSTLY_Q4_0:     return "Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1:     return "Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q5_0:     return "Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1:     return "Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q6_0:     return "Q6_0";
        case LLAMA_FTYPE_MOSTLY_Q8_0:     return "Q8_0";
        case LLAMA_FTYPE_MOSTLY_Q8_KV:    return "Q8_KV";
        case LLAMA_FTYPE_MOSTLY_Q2_K:     return "Q2_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q2_K_R4:  return "Q2_K_R4";
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:   return "Q2_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:   return "Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:   return "Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:   return "Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q3_K_R4:  return "Q3_K_R4";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:   return "Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_R4:  return "Q4_K_R4";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:   return "Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:   return "Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_R4:  return "Q5_K_R4";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:   return "Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K:     return "Q6_K";
        case LLAMA_FTYPE_MOSTLY_Q6_K_R4:  return "Q6_K_R4";
        case LLAMA_FTYPE_MOSTLY_Q8_K_R8:  return "Q8_K_R8";
        case LLAMA_FTYPE_MOSTLY_Q8_K_R16: return "Q8_K_R16";
        case LLAMA_FTYPE_MOSTLY_Q8_KV_R8: return "Q8_KV_R8";
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:  return "IQ2_XXS - 2.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4:return "IQ2_XXS_R4 - 2.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:   return "IQ2_XS - 2.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XS_R4:return "IQ2_XS_R4 - 2.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_KS:   return "IQ2_KS - 2.1875 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_S:    return "IQ2_S - 2.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_M:    return "IQ2_M - 2.7 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_M_R4: return "IQ2_M_R4 - 2.7 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:   return "IQ3_XS - 3.3 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:  return "IQ3_XXS - 3.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_KT:   return "IQ1_KT - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_PQ2_0:    return "PQ2_0 - 2.13 bpw ternary (Prism)";
        // ik 77eff8798 (#2596): the repacked Prism and IQ4_KS_R16 ftypes had no name, so the loader ignored a declared
        // general.file_type of 161 / 162 / 346 (ik #2550 rule) and printed a "(guessed)" content type instead
        case LLAMA_FTYPE_MOSTLY_PQ2_0_R8: return "PQ2_0_R8 - 2.13 bpw ternary (Prism)";
        case LLAMA_FTYPE_MOSTLY_PTQ1_0:   return "PTQ1_0 - 1.75 bpw ternary (Prism)";
        case LLAMA_FTYPE_MOSTLY_PTQ1_0_R8:return "PTQ1_0_R8 - 1.75 bpw ternary (Prism)";
        // mainline's names (ggml-org/llama.cpp src/llama-model-loader.cpp llama_model_ftype_name, MIT)
        case LLAMA_FTYPE_MOSTLY_Q1_0:     return "Q1_0";
        case LLAMA_FTYPE_MOSTLY_Q2_0:     return "Q2_0";
        case LLAMA_FTYPE_MOSTLY_F8_E4M3:  return "F8_E4M3";
        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE:return "MXFP4 MoE";
        case LLAMA_FTYPE_MOSTLY_NVFP4:    return "NVFP4";
        case LLAMA_FTYPE_MOSTLY_TQK6:     return "TQK6 - 1.625 bpw trellis (group 128)";
        case LLAMA_FTYPE_MOSTLY_TQK7:     return "TQK7 - 1.875 bpw trellis (group 128)";
        case LLAMA_FTYPE_MOSTLY_TQ2_T:    return "TQ2_T - 2.125 bpw trellis (group 128)";
        case LLAMA_FTYPE_MOSTLY_TQP5:     return "TQP5 - 2.625 bpw pair trellis (group 128)";
        case LLAMA_FTYPE_MOSTLY_TQP6:     return "TQP6 - 3.125 bpw pair trellis (group 128)";
        case LLAMA_FTYPE_MOSTLY_TQP7:     return "TQP7 - 3.625 bpw pair trellis (group 128)";
        case LLAMA_FTYPE_MOSTLY_IQ2_KT:   return "IQ2_KT - 2.125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_KT:   return "IQ3_KT - 3.125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_KT:   return "IQ4_KT - 4.0 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4: return "IQ3_XXS_R4 - 3.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_S:    return "IQ1_S - 1.5625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_S_R4: return "IQ1_S_R4 - 1.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_M_R4: return "IQ1_M_R4 - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_M:    return "IQ1_M - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_XS:   return "IQ1_XS - 1.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_XXS:  return "IQ1_XXS - 1.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_XXXS: return "IQ1_XXXS - 1.1875 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:   return "IQ4_NL - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_NL_R4:return "IQ4_NL_R4 - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_XS_R8:return "IQ4_XS_R8 - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_Q4_0_R8:  return "Q4_0_R8 - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_Q2_0_R8:  return "Q2_0_R8 - 2.25 bpw";
        case LLAMA_FTYPE_MOSTLY_Q5_0_R4:  return "Q5_0_R4 - 5.5 bpw";
        case LLAMA_FTYPE_MOSTLY_Q6_0_R4:  return "Q6_0_R4 - 6.5 bpw";
        case LLAMA_FTYPE_MOSTLY_Q8_0_R8:  return "Q8_0_R8 - 8.5 bpw";
        case LLAMA_FTYPE_MOSTLY_MXFP4:    return "MXFP4 - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_MXFP6_E2M3: return "MXFP6_E2M3 - 6.25 bpw";
        case LLAMA_FTYPE_MOSTLY_MXFP6_E3M2: return "MXFP6_E3M2 - 6.25 bpw";
        case LLAMA_FTYPE_MOSTLY_MXFP8_E4M3: return "MXFP8_E4M3 - 8.25 bpw";
        case LLAMA_FTYPE_MOSTLY_MXFP8_E5M2: return "MXFP8_E5M2 - 8.25 bpw";
        case LLAMA_FTYPE_MOSTLY_TQ3_1S:     return "TQ3_1S - 4.0 bpw WHT-rotated";
        case LLAMA_FTYPE_MOSTLY_TQ4_1S:     return "TQ4_1S - 5.0 bpw WHT-rotated";
        // MXFP4_R8 had a quantize mapping (llama-quantize.cpp) and an entry in the quantize CLI
        // table ("MXFP4_R8", " MXFP4 repacked") but NO name here, so a model quantised to it
        // reported itself as "unknown, may not work". Found by lint_ftype_case_coverage.py on
        // its first correct run -- pre-existing, and the same defect class as Q2_0_G128 above.
        // A repack carries the bpw of its base type, per Q8_0_R8 - 8.5 bpw two lines up.
        case LLAMA_FTYPE_MOSTLY_MXFP4_R8: return "MXFP4_R8 - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_NVFP4_KS: return "NVFP4_KS - 4.5 bpw"; // (NVFP4 itself is named above)
        case LLAMA_FTYPE_MOSTLY_Q1_0_G128:return "Q1_0_G128 - 1.125 bpw";
        case LLAMA_FTYPE_MOSTLY_Q2_0_G128:return "Q2_0_G128 - 2.125 bpw";
        case LLAMA_FTYPE_MOSTLY_Q1_0_G128_R8:return "Q1_0_G128_R8";
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:   return "IQ4_XS - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_KS:   return "IQ4_KS - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_KS_R4:return "IQ4_KS_R4 - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_KS_R16:return "IQ3_KS_R16 - 3.1875 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_KS_R16:return "IQ4_KS_R16 - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ5_KS_R4:return "IQ5_KS_R4 - 5.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ5_KS_R16:return "IQ5_KS_R16 - 5.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_KSS:  return "IQ4_KSS - 4.0 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ5_KS:   return "IQ5_KS - 5.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_K:    return "IQ2_K - 2.375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_K_R4: return "IQ2_K_R4 - 2.375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_KS:   return "IQ3_KS - 3.1875 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_KL:   return "IQ2_KL - 2.6875 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_K:    return "IQ3_K - 3.4325 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_K_R4: return "IQ3_K_R4 - 3.4325 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_KL:   return "IQ3_KL - 4 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_K:    return "IQ4_K - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_K_R4: return "IQ4_K_R4 - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ5_K:    return "IQ5_K - 5.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ5_K_R4: return "IQ5_K_R4 - 5.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ6_K:    return "IQ6_K - 6.6 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_BN:   return "IQ1_BN - 1.625 bpw Bitnet";
        case LLAMA_FTYPE_MOSTLY_IQ2_BN:   return "IQ2_BN - 2.00 bpw Bitnet";
        case LLAMA_FTYPE_MOSTLY_IQ2_BN_R4:return "IQ2_BN_R4 - 2.00 bpw Bitnet";
        case LLAMA_FTYPE_MOSTLY_IQ3_S:    return "IQ3_S - 3.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_S_R4: return "IQ3_S_R4 - 3.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_M:    return "IQ3_S mix - 3.66 bpw";
        case LLAMA_FTYPE_MOSTLY_Q4_0_4_4: return "Q4_0_4_4";
        case LLAMA_FTYPE_MOSTLY_Q4_0_4_8: return "Q4_0_4_8";
        case LLAMA_FTYPE_MOSTLY_Q4_0_8_8: return "Q4_0_8_8";

        default: return "unknown, may not work";
    }
}

const char * llama_model_type_name(e_model type) {
    switch (type) {
        case MODEL_14M:           return "14M";
        case MODEL_17M:           return "17M";
        case MODEL_22M:           return "22M";
        case MODEL_33M:           return "33M";
        case MODEL_60M:           return "60M";
        case MODEL_70M:           return "70M";
        case MODEL_80M:           return "80M";
        case MODEL_109M:          return "109M";
        case MODEL_137M:          return "137M";
        case MODEL_140M:          return "140M";
        case MODEL_160M:          return "160M";
        case MODEL_190M:          return "190M";
        case MODEL_220M:          return "220M";
        case MODEL_250M:          return "250M";
        case MODEL_256M:          return "256M";
        case MODEL_270M:          return "270M";
        case MODEL_335M:          return "335M";
        case MODEL_350M:          return "350M";
        case MODEL_360M:          return "360M";
        case MODEL_410M:          return "410M";
        case MODEL_450M:          return "450M";
        case MODEL_475M:          return "475M";
        case MODEL_558M:          return "558M";
        case MODEL_700M:          return "700M";
        case MODEL_770M:          return "770M";
        case MODEL_780M:          return "780M";
        case MODEL_950M:          return "950M";
        case MODEL_0_3B:          return "0.3B";
        case MODEL_0_5B:          return "0.5B";
        case MODEL_0_6B:          return "0.6B";
        case MODEL_0_8B:          return "0.8B";
        case MODEL_1B:            return "1B";
        case MODEL_1_2B:          return "1.2B";
        case MODEL_1_3B:          return "1.3B";
        case MODEL_1_4B:          return "1.4B";
        case MODEL_1_5B:          return "1.5B";
        case MODEL_1_6B:          return "1.6B";
        case MODEL_1_7B:          return "1.7B";
        case MODEL_1_8B:          return "1.8B";
        case MODEL_2B:            return "2B";
        case MODEL_2_6B:          return "2.6B";
        case MODEL_2_8B:          return "2.8B";
        case MODEL_2_9B:          return "2.9B";
        case MODEL_3B:            return "3B";
        case MODEL_4B:            return "4B";
        case MODEL_6B:            return "6B";
        case MODEL_6_9B:          return "6.9B";
        case MODEL_7B:            return "7B";
        case MODEL_8B:            return "8B";
        case MODEL_9B:            return "9B";
        case MODEL_11B:           return "11B";
        case MODEL_12B:           return "12B";
        case MODEL_13B:           return "13B";
        case MODEL_14B:           return "14B";
        case MODEL_15B:           return "15B";
        case MODEL_16B:           return "16B";
        case MODEL_20B:           return "20B";
        case MODEL_26B_A4B:       return "26B.A4B";
        case MODEL_27B:           return "27B";
        case MODEL_30B:           return "30B";
        case MODEL_32B:           return "32B";
        case MODEL_34B:           return "34B";
        case MODEL_35B:           return "35B";
        case MODEL_36B:           return "36B";
        case MODEL_36B_A4B:       return "36B.A4B";
        case MODEL_40B:           return "40B";
        case MODEL_65B:           return "65B";
        case MODEL_70B:           return "70B";
        case MODEL_120B:          return "120B";
        case MODEL_142B:          return "142B";
        case MODEL_236B:          return "236B";
        case MODEL_290B:          return "290B";
        case MODEL_314B:          return "314B";
        case MODEL_405B:          return "405B";
        case MODEL_671B:          return "671B";
        case MODEL_SMALL:         return "0.1B";
        case MODEL_MEDIUM:        return "0.4B";
        case MODEL_LARGE:         return "0.8B";
        case MODEL_XL:            return "1.5B";
        case MODEL_A1_7B:         return "A1.7B";
        case MODEL_A2_7B:         return "A2.7B";
        case MODEL_8x7B:          return "8x7B";
        case MODEL_8x22B:         return "8x22B";
        case MODEL_16x12B:        return "16x12B";
        case MODEL_16x3_8B:       return "16x3.8B";
        case MODEL_10B_128x3_66B: return "10B+128x3.66B";
        case MODEL_57B_A14B:      return "57B.A14B";
        case MODEL_17B_16E:       return "17Bx16E (Scout)";
        case MODEL_17B_128E:      return "17Bx128E (Maverick)";
        case MODEL_A13B:          return "A13B";
        case MODEL_7B_A1B:        return "7B.A1B";
        case MODEL_8B_A1B:        return "8B.A1B";
        case MODEL_12B_A2_5B:     return "12B.A2.5B";
        case MODEL_16B_A1B:       return "16B.A1B";
        case MODEL_21B_A3B:       return "21B.A3B";
        case MODEL_30B_A3B:       return "30B.A3B";
        case MODEL_33B_A3B:       return "33B.A3B";
        case MODEL_35B_A3B:       return "35B.A3B";
        case MODEL_80B_A3B:       return "80B.A3B";
        case MODEL_125B_A6B:      return "125B.A6B";
        case MODEL_80B_A13B:      return "80B.A13B";
        case MODEL_100B_A6B:      return "100B.A6B";
        case MODEL_106B_A12B:     return "106B.A12B";
        case MODEL_119B_A6B:      return "119B.A6B";
        case MODEL_122B_A10B:     return "122B.A10B";
        case MODEL_230B_A10B:     return "230B.A10B";
        case MODEL_235B_A22B:     return "235B.A22B";
        case MODEL_310B_A15B:     return "310B.A15B";
        case MODEL_312B_A17B:     return "312B.A17B";
        case MODEL_300B_A47B:     return "300B.A47B";
        case MODEL_355B_A32B:     return "355B.A32B";
        case MODEL_397B_A17B:     return "397B.A17B";
        case MODEL_744B_A40B:     return "744B.A40B";
        case MODEL_E2B:           return "E2B";
        case MODEL_E4B:           return "E4B";
        case LLM_TYPE_816B_A42B:     return "816B.A42B";
        case LLM_TYPE_78B_A3_5B:     return "78B.A3.5B";
        case LLM_TYPE_1B_A400M:      return "1B.A400M";
        case LLM_TYPE_3B_A800M:      return "3B.A800M";
        case LLM_TYPE_32B_A9B:       return "32B.A9B";
        case LLM_TYPE_75B_A9B:       return "75B.A9B";
        case LLM_TYPE_A3B:           return "A3B";
        case LLM_TYPE_320B_A18B:     return "320B.A18B";
        default:                  return "?B";
    }
}

// ---- DiffusionGemma public API ---------------------------------------------------------------
// All of these are per-MODEL rather than per-context because the diffusion state (self-conditioning
// buffers, prompt-KV store) is device memory owned by the model, exactly as upstream had it on the
// llama_model_diffusion_gemma subclass this fork's plain-struct llama_model cannot express.

bool llama_model_is_diffusion(const llama_model * model) {
    return model && model->arch == LLM_ARCH_DIFFUSION_GEMMA;
}

void llama_diffusion_set_sc(llama_model * model, const float * sc_logits,
                            float use_sc, float temp_inv, bool enabled) {
    if (!llama_model_is_diffusion(model)) {
        return;
    }
    model->dg_sc_logits_ptr = sc_logits;
    model->dg_sc_use        = use_sc;
    model->dg_sc_temp_inv   = temp_inv;
    model->dg_sc_enabled    = enabled;
}

void llama_diffusion_set_device_sc(llama_model * model, bool enabled) {
    if (!llama_model_is_diffusion(model)) {
        return;
    }
    model->dg_sc_device_resident = enabled;
}

void llama_diffusion_set_phase(llama_model * model, int phase, int32_t P, int32_t off) {
    if (!llama_model_is_diffusion(model)) {
        return;
    }
    model->dg_pkv_phase       = phase;
    model->dg_pkv_P           = P;
    model->dg_pkv_prefill_off = off;   // PREFILL chunk start; ignored by UNIFIED/DECODE
}

static const std::map<llama_rope_section_order, const char *> LLAMA_ROPE_SECTION_ORDERS = {
    { LLAMA_ROPE_SECTION_ORDER_TYXZ, "tyxz" },
    { LLAMA_ROPE_SECTION_ORDER_ZYXT, "zyxt" },
};

std::string llama_rope_section_order_name(llama_rope_section_order rope_section_order) {
    return LLAMA_ROPE_SECTION_ORDERS.at(rope_section_order);
}

size_t llama_diffusion_pkv_bytes_per_token(const llama_model * model, bool use_f16) {
    if (!llama_model_is_diffusion(model)) {
        return 0;
    }
    const size_t elt = use_f16 ? sizeof(ggml_fp16_t) : sizeof(float);
    size_t bytes = 0;
    for (int il = 0; il < (int) model->hparams.n_layer; ++il) {
        bytes += (size_t) model->hparams.n_embd_head_k(il) * model->hparams.n_head_kv(il) * 2 * elt; // K + V
    }
    return bytes;
}

bool llama_diffusion_device_sample(const llama_model * model, const float * u, int * argmax,
                                   float * entropy, int * sampled, int n_tokens, float inv_temp) {
    if (!llama_model_is_diffusion(model) || model->dg_sc_dev == nullptr ||
        !u || !argmax || !entropy || !sampled || n_tokens <= 0) {
        return false;
    }
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return ggml_backend_cuda_diffusion_sample(model->dg_sc_dev, u, argmax, entropy, sampled,
                                              n_tokens, inv_temp);
#else
    // no device sampler in this build -- the caller falls back to the host reduction path
    return false;
#endif
}
// ^ RESTORED 2026-09-08. llama_diffusion_device_sample's closing brace was missing: the merge
// appended the upstream accessor block below straight after this function's #endif without ever
// closing the function, so every accessor that follows was being parsed INSIDE it --
// `error C2601: 'llama_model_free': local function definitions are illegal`, 95 of them.
// Same shape as the stray-closer defect in src/llama-model.h: an adopted block pasted at the wrong
// side of a brace. Note the file still BALANCED on a naive count for a while, and one of my own
// tools reported it balanced because it clamped negative depth to zero -- the honest instrument is
// "where does top-level depth stop returning to 0", which pointed straight here.

// PRIOR-ART-NONE: searched already-resolved src/*.cpp (loader.cpp, llama.cpp, llama-quant.cpp,
// llama-hparams.cpp, llama-load-tensors.cpp) for an existing definition of each function below
// before concluding upstream's model.cpp side is the only source; none found.
// ---- adopted from upstream in this merge: public API accessor functions declared LLAMA_API in
// include/llama.h with NO definition anywhere else in src/ (measured: scripted scan of every
// LLAMA_API name in llama.h against every already-resolved .cpp file; 0 hits before this). All
// are simple field/switch accessors on `model->arch` / `model->hparams` / `model->classifier_labels`
// / `model->gguf_kv`, which this fork's plain-struct llama_model already has as data members, so
// each ports with no logic change. The ONE exception is llama_model_ftype: upstream's body called
// `model->ftype()` (a METHOD, from the excluded polymorphic llama_model_base class); adapted to
// `model->ftype` (a plain DATA MEMBER on this fork's struct) -- caught by reading the body against
// the actual struct shape, not by assuming a pure textual port would compile.
void llama_model_free(llama_model * model) {
    delete model;
}

int32_t llama_model_n_ctx_train(const llama_model * model) {
    return model->hparams.n_ctx_train;
}

// Adapted 2026-09-08, same shape as the llama_model_ftype adaptation noted above: upstream's
// bodies call hparams.n_embd_out() and hparams.n_layer() as METHODS, because upstream's
// llama_hparams exposes them that way. This fork declares both as plain DATA MEMBERS
// (src/llama-hparams.h:28 and :29), so the parentheses come off. Caught by reading each body
// against the actual struct rather than assuming a textual port would compile --
// "term does not evaluate to a function taking 0 arguments" is what it says otherwise.
int32_t llama_model_n_embd_out(const llama_model * model) {
    return model->hparams.n_embd_out;
}

int32_t llama_model_n_layer(const llama_model * model) {
    return model->hparams.n_layer;
}

int32_t llama_model_n_layer_nextn(const llama_model * model) {
    return model->hparams.n_layer_nextn;
}

// Accessors declared in src/llama-ext.h and previously defined nowhere in this tree, so every
// caller was an unresolved symbol at link. Bodies taken from upstream's src/llama-model.cpp; they
// compile against this fork's model unchanged because they touch only members it already has:
// `std::vector<int32_t> devices` (llama-model.h:925), `std::vector<int32_t> target_layer_ids`
// (:958) and hparams.n_expert.
//
// GATE-CLAIM-OK: nothing below claims a device is unavailable or busy. Every "CPU"/"GPU" here
// names a ggml_backend_dev_type enumerator or a compile-time backend selection, read out of
// ggml/include/ggml-backend.h:136-144; no hardware state is asserted and none is needed.
//
// llama_model_get_device IS now implemented, below, after the earlier "left undefined on purpose"
// note was re-examined 2026-09-08 and found to be right about the HAZARD and wrong about the
// IMPOSSIBILITY. Its reasoning is preserved because the hazard is real: upstream implements this
// as `return model->devices[i].dev;` over a std::vector<llama_device> of device HANDLES, while
// this fork's `devices` is std::vector<int32_t> of device ORDINALS -- MEASURED at
// src/llama.cpp:597, where the value goes straight into
// ggml_backend_cuda_get_device_memory(device, ...) and, past the local count, indexes
// model.rpc_servers[]. So `ggml_backend_dev_get(model->devices[i])` would compile and silently
// address the WRONG device, and a nullptr stub would let common/fit.cpp build and then select no
// device at all. Both are worse than a link error, and neither is what is written here.
//
// WHAT MADE IT POSSIBLE: ggml keeps a device REGISTRY (ggml/src/ggml-backend-reg.cpp:203, :302
// populate it) which this fork's built src/ path never consults -- `ggml_backend_dev_count` and
// `ggml_backend_dev_get` have zero callers across src/llama.cpp, llama-model.cpp,
// llama-load-tensors.cpp and llama-build-context.cpp, because the fork selects its backend at
// COMPILE time via #if defined(GGML_USE_CUDA). The registry is populated regardless, so the
// ordinal can be resolved to a handle rather than declared unresolvable.
//
// THE ASSUMPTION, stated because it is the thing that could be wrong: the fork's ordinal N is the
// N-th non-CPU device in registry enumeration order. That holds for this fork's compile-time
// single-backend selection, where the registry lists CUDA0, CUDA1, ... in CUDA ordinal order. It
// would NOT hold in a build registering two accelerator backends at once (CUDA and Vulkan
// together), where registry order interleaves vendors. Rather than guess in that case, the loop
// below returns nullptr when the ordinal cannot be resolved -- the caller sees "no such device",
// which is checkable, instead of a confidently wrong handle.
ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i) {
    if (model == nullptr || i < 0 || i >= (int) model->devices.size()) {
        return nullptr;
    }

    const int ordinal = model->devices[i];
    if (ordinal < 0) {
        return nullptr;
    }

    int seen = 0;
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t d = 0; d < n_dev; ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        if (dev == nullptr) {
            continue;
        }
        // The host and META entries are not what `devices` enumerates -- that vector holds the
        // offload targets, which is why llama_get_device_memory() hands its value to an
        // accelerator backend.
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        if (t == GGML_BACKEND_DEVICE_TYPE_CPU || t == GGML_BACKEND_DEVICE_TYPE_META) {
            continue;
        }
        if (seen == ordinal) {
            return dev;
        }
        ++seen;
    }

    // Ordinal past the end of the registry's accelerator devices. Reachable for an RPC entry,
    // which src/llama.cpp:597-608 addresses through model.rpc_servers[] rather than through any
    // local device, and which therefore has no registry handle to return.
    return nullptr;
}

bool llama_prec_policy::apply(ggml_tensor * res) const {
    if (!res || !res->src[0]) {
        return false;
    }

    const auto it = prec_src1.find(res->src[0]);
    if (it == prec_src1.end()) {
        return false;
    }

    return ggml_prec_set_src(res, it->second, 1);
}

void llama_prec_policy::load(llama_model_loader & ml, const llama_model & model) {
    std::vector<std::string> tensor_names;
    if (!ml.get_arr(LLM_KV_GENERAL_TENSOR_EXTRA_NAME, tensor_names, false)) {
        return;
    }

    const gguf_context * ctx = ml.metadata;
    const std::string key = ml.llm_kv(LLM_KV_GENERAL_TENSOR_EXTRA_PREC_A4);
    const int kid = gguf_find_key(ctx, key.c_str());
    if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, kid) != GGUF_TYPE_BOOL) {
        throw std::runtime_error(format("%s must be a bool array", key.c_str()));
    }

    const size_t n_values = gguf_get_arr_n(ctx, kid);
    if (n_values != tensor_names.size()) {
        throw std::runtime_error(format(
            "%s tensor/value length mismatch (%zu vs %zu)",
            key.c_str(), tensor_names.size(), n_values));
    }

    // tensors that can not use 4-bit activations, keep src1 at higher precision
    const int8_t * values = (const int8_t *) gguf_get_arr_data(ctx, kid);
    std::unordered_set<std::string> want;
    for (size_t i = 0; i < n_values; ++i) {
        if (values[i] == 0) {
            want.insert(tensor_names[i]);
        }
    }

    // resolve names to tensor pointers
    for (const auto & [name, w] : model.tensors_by_name) {
        if (want.count(name)) {
            prec_src1.emplace(w, GGML_PREC_Q8);
        }
    }
}

int32_t llama_model_n_devices(const struct llama_model * model) {
    return (int32_t)model->devices.size();
}

int32_t llama_model_n_expert(const struct llama_model * model) {
    return model->hparams.n_expert;
}

const int32_t * llama_model_target_layer_ids(const struct llama_model * model) {
    const auto & v = model->target_layer_ids;
    return v.empty() ? nullptr : v.data();
}

uint32_t llama_model_target_layer_ids_n(const struct llama_model * model) {
    return (uint32_t) model->target_layer_ids.size();
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

int32_t llama_model_n_head_kv(const llama_model * model) {
    return model->hparams.n_head_kv();
}

int32_t llama_model_n_swa(const llama_model * model) {
    // dsv4 kv-cache has SWA but it cannot be used as a rollback because of
    // other compression ratios, so we return 0 here
    if (llm_arch_is_dsv4(model->arch)) {
        return 0;
    }
    return model->hparams.n_swa;
}

uint32_t llama_model_n_cls_out(const struct llama_model * model) {
    return model->hparams.n_cls_out;
}

const char * llama_model_cls_label(const struct llama_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }
    return nullptr;
}

const char * llama_model_decision_label(const struct llama_model * model, uint32_t i) {
    if (i < model->decision_labels.size()) {
        return model->decision_labels[i].c_str();
    }

    return nullptr;
}

// deprecated
int32_t llama_n_ctx_train(const llama_model * model) {
    return llama_model_n_ctx_train(model);
}

// deprecated
int32_t llama_n_embd(const llama_model * model) {
    return llama_model_n_embd(model);
}

// deprecated
int32_t llama_n_layer(const llama_model * model) {
    return llama_model_n_layer(model);
}

// deprecated
int32_t llama_n_head(const llama_model * model) {
    return llama_model_n_head(model);
}

// `enum llama_rope_type`, not the bare tag. This fork also declares a FUNCTION named
// llama_rope_type (include/llama.h:917), and in C a tag and an ordinary identifier live in
// separate namespaces -- so the bare name here resolves to the FUNCTION, and the compiler reports
// "error C2373: 'llama_rope_type': redefinition; different type modifiers" pointing at a line that
// looks fine. The `enum` keyword disambiguates it. include/llama.h:952 already spells the
// declaration this way; only the definition was missing it.
enum llama_rope_type llama_model_rope_type(const llama_model * model) {
    // Return exactly what inference uses. This was a stale copy of upstream's switch with no case
    // for QWEN35 / QWEN35MOE / QWEN3VL / QWEN3VLMOE / QWEN4EXP (or any arch added since), so they
    // fell to `default: NORM`. Its comment called that "display-only" -- it is NOT: mtmd reads it to
    // pick the decoder position layout (mtmd.cpp pos_type, mtmd-helper-gen.cpp mrope). mtmd then
    // built 1 pos per image token while llama_decode_internal, keyed on hparams.rope_type (IMROPE),
    // read 4 sections -> over-read of the pos vector: garbage h/w/extra positions (run-to-run
    // nondeterministic output) or 0xC0000005 when the read crossed a page (Qwen3.8-27B + mmproj,
    // 3 of 6 plain runs). hparams.rope_type is set from llama_rope_type() at load
    // (llama-hparams.cpp), the fork's maintained switch, so both sides now agree by construction.
    return model->hparams.rope_type;
}

float llama_model_rope_freq_scale_train(const llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

const char * llama_model_meta_key_str(llama_model_meta_key key) {
    switch (key) {
        case LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case LLAMA_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
}

// adapted: upstream called `model->ftype()` (a method on the excluded polymorphic class);
// this fork's llama_model declares `ftype` as a plain data member, so this reads the field.
llama_ftype llama_model_ftype(const llama_model * model) {
    return model->ftype;
}
// A STRAY `}` STOOD HERE and was removed 2026-09-08. It is the other half of the missing brace on
// llama_diffusion_device_sample ~250 lines above: the adopted upstream accessor block was pasted
// into that function's body, and this extra closer at the far end made the FILE BALANCE. That is
// why a brace count called the file clean while the compiler reported 95 'local function
// definitions are illegal' -- the two defects cancel in a total and do not cancel in a parse.
// Fixing only one of them makes the count WORSE while making the file more correct, which is
// exactly what happened here and is worth remembering before trusting a balance check.

// SOURCE-READ: tmp/_unsl-branches_g3.out (git show 4e1865e3 of the unslothai/llama.cpp remote, MIT,
// llama_model::prefetch_rows); the mapping-level half is already in src/llama-mmap.cpp (llama_mmap::prefetch_rows).
void llama_model::prefetch_rows(const struct ggml_tensor * t, const int32_t * rows, size_t n_rows) const {
    if (t == nullptr || t->data == nullptr || rows == nullptr || n_rows == 0) {
        return;
    }

    for (const auto & mapping : mappings) {
        if (mapping->contains(t->data, ggml_nbytes(t))) {
            mapping->prefetch_rows(t->data, t->nb[1], ggml_row_size(t->type, t->ne[0]), rows, n_rows);
            return;
        }
    }
}

bool llama_model_is_recurrent(const llama_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_hybrid(const llama_model * model) {
    return llm_arch_is_hybrid(model->arch);
}

bool llama_model_has_recurrent(const llama_model * model) {
    return llm_arch_is_hybrid(model->arch) || llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_deepseek4(const llama_model * model) {
    return model && llm_arch_is_dsv4(model->arch);
}

bool llama_model_is_openpangu(const llama_model * model) {
    return model && model->arch == LLM_ARCH_OPENPANGU;
}

bool llama_model_is_gemma4_mtp_assistant(const llama_model * model) {
    return model && (model->arch == LLM_ARCH_GEMMA4_MTP || model->arch == LLM_ARCH_GEMMA4_ASSISTANT);
}

bool llama_model_is_step35(const llama_model * model) {
    return model && model->arch == LLM_ARCH_STEP35;
}

bool llama_model_is_qwen4exp(const llama_model * model) {
    return model && model->arch == LLM_ARCH_QWEN4EXP;
}

// qwen4exp marks a present trunk block with hc_attn_norm, every other arch with attn_norm
static bool llama_model_trunk_block_present(const llama_layer & layer) {
    return layer.attn_norm != nullptr || layer.hc_attn_norm != nullptr;
}

bool llama_model_is_qwen35_family(const llama_model * model) {
    return model && (model->arch == LLM_ARCH_QWEN35 || model->arch == LLM_ARCH_QWEN35MOE);
}

enum llama_mtp_package llama_model_mtp_package(const llama_model * model) {
    if (!model) {
        return LLAMA_MTP_PACKAGE_INVALID;
    }

    if (llama_model_is_gemma4_mtp_assistant(model) && model->hparams.nextn_predict_layers == 0) {
        return LLAMA_MTP_PACKAGE_COMPANION;
    }

    const size_t n_nextn = model->hparams.nextn_predict_layers;
    const bool has_common_package_contract =
        llama_model_is_step35(model) || llama_model_is_deepseek4(model) ||
        llama_model_is_qwen35_family(model) || llama_model_is_qwen4exp(model) ||
        llama_model_is_gemma4_mtp_assistant(model);
    if (!has_common_package_contract) {
        return n_nextn > 0 ? LLAMA_MTP_PACKAGE_EMBEDDED : LLAMA_MTP_PACKAGE_NONE;
    }

    if (n_nextn == 0) {
        if (llama_model_is_step35(model) || llama_model_is_deepseek4(model) ||
            llama_model_is_qwen35_family(model) || llama_model_is_qwen4exp(model)) {
            for (const auto & layer : model->layers) {
                if (llama_model_trunk_block_present(layer)) {
                    return LLAMA_MTP_PACKAGE_TARGET_ONLY;
                }
            }
        }
        return LLAMA_MTP_PACKAGE_NONE;
    }

    const size_t n_layers = model->layers.size();
    if (n_nextn > n_layers) {
        return LLAMA_MTP_PACKAGE_INVALID;
    }

    const size_t first = n_layers - n_nextn;
    // A Qwen3.5 NextN block loads eh_proj, attn_q and the MLP as optional, so none of them
    // marks a predictor tail on its own; enorm is required and is present in all of them.
    const bool has_tail = model->layers[first].nextn.eh_proj != nullptr ||
        (llama_model_is_qwen35_family(model) && model->layers[first].nextn.enorm != nullptr);

    bool has_trunk = false;
    for (size_t il = 0; il < first; ++il) {
        if (llama_model_trunk_block_present(model->layers[il])) {
            has_trunk = true;
            break;
        }
    }

    if (has_trunk && has_tail) {
        return LLAMA_MTP_PACKAGE_EMBEDDED;
    }
    if (has_trunk && !has_tail) {
        return LLAMA_MTP_PACKAGE_TARGET_ONLY;
    }
    if (!has_trunk && has_tail) {
        return LLAMA_MTP_PACKAGE_COMPANION;
    }
    return LLAMA_MTP_PACKAGE_INVALID;
}

bool llama_model_step35_has_nextn_weights(const llama_model * model) {
    if (!model || !llama_model_is_step35(model) || model->hparams.nextn_predict_layers == 0) {
        return false;
    }

    const size_t n_layers = model->layers.size();
    const size_t n_nextn = model->hparams.nextn_predict_layers;
    if (n_nextn > n_layers) {
        return false;
    }

    const size_t first = n_layers - n_nextn;
    for (size_t il = first; il < n_layers; ++il) {
        const llama_layer & layer = model->layers[il];
        const llama_layer_nextn & nextn = layer.nextn;
        if (!nextn.eh_proj || !nextn.enorm || !nextn.hnorm ||
            !layer.attn_norm || !layer.wq || !layer.wk || !layer.wv || !layer.wo ||
            !layer.ffn_norm || !layer.ffn_gate || !layer.ffn_up || !layer.ffn_down) {
            return false;
        }
    }
    return true;
}

bool llama_is_gemma4_mtp_file(const char * path) {
    if (!path || !*path) return false;
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * ctx = gguf_init_from_file(path, params);
    if (!ctx) return false;
    bool result = false;
    const int key_id = gguf_find_key(ctx, "general.architecture");
    if (key_id >= 0) {
        const char * arch = gguf_get_val_str(ctx, key_id);
        if (arch && strcmp(arch, "gemma4_mtp") == 0) {
            result = true;
        }
    }
    gguf_free(ctx);
    return result;
}

bool llama_model_is_split_mode_graph(const struct llama_model * model) {
    return model && (model->split_mode == LLAMA_SPLIT_MODE_GRAPH || model->split_mode == LLAMA_SPLIT_MODE_ATTN);
}

bool llama_model_supports_ctx_shift(const struct llama_model * model) {
    // openPangu and DeepSeek4 keep position-dependent private state outside the generic KV cache.
    // Gemma3 and Cohere2 vary rope geometry per layer without recording it in hparams.swa_layers.
    return model && model->arch != LLM_ARCH_OPENPANGU && !llm_arch_is_dsv4(model->arch)
        && model->arch != LLM_ARCH_GEMMA3 && model->arch != LLM_ARCH_COHERE2;
}

bool llama_model_supports_partial_kv_reuse(const struct llama_model * model) {
    return model != nullptr;
}

llm_tensor llm_tensor_type(llm_arch arch, const std::string & tensor_name, int il) {
    const auto & names = llm_tensor_names_for(arch);
    if (il < 0) {
        for (auto & entry : names) {
            if (tensor_name.find(entry.second) == 0) {
                return entry.first;
            }
        }
        return LLM_TENSOR_UNKNOWN;
    }
    for (auto & entry : names) {
        auto base_name = ::format(entry.second.c_str(), il);
        auto this_name = base_name + ".weight";
        if (tensor_name == this_name) {
            return entry.first;
        }
        this_name = base_name + ".bias";
        if (tensor_name == this_name) {
            return entry.first;
        }
        if (tensor_name == base_name) {
            return entry.first;
        }
    }
    return LLM_TENSOR_UNKNOWN;
}

size_t llama_model::cache_size(int il, ggml_type type_k, ggml_type type_v, ggml_type idx_type_k, uint32_t kv_size, int mla_attn, int n_seq_max, bool flash_attn,
                               bool swa_compress, uint32_t n_ubatch) const {
    if (il < 0 || il >= (int) hparams.n_layer) return 0;
    if (hparams.is_recr_impl[il]) {
        auto state_sots = std::min<uint32_t>(std::max<uint32_t>(1, n_seq_max), kv_size);
        // upstream's Kimi KDA and MiniMax linear-attention families (their class loaders set n_embd_head_kda / _la) are sized by
        // upstream's n_embd_r() + n_embd_s(). The ssm_n_group branch is the fork's qwen3next delta-net layout and divides by
        // dt_rank / n_group, which is 0 for a KDA model (test-llama-archs glm5-next: 0xC0000094 here). Other families unchanged.
        const bool kda_or_la = hparams.n_embd_head_kda != 0 || hparams.n_embd_head_la != 0;
        const size_t state_width = kda_or_la
            ? (size_t) hparams.n_embd_r() + hparams.n_embd_s() + hparams.n_embd_ple_conv((uint32_t) il)
            : hparams.ssm_n_group > 0
            ? hparams.recurrent_state_layout_for((uint32_t) il).row_width
            : hparams.n_embd_v_s() + hparams.n_embd_ple_conv((uint32_t) il);
        return state_width * (size_t) state_sots * sizeof(float);
    }
    if (arch == LLM_ARCH_OPENPANGU) {
        // MLA-latent cache: K row [ckv | roped k_pe]. The value-side latent is
        // rederived from K per graph. DSA layers also cache one indexer key per
        // position. The recurrent conv slot is constant-size and negligible here.
        // openPangu forces flash_attn off, so the pad is always the non-FA one
        const uint32_t pad = llama_kv_cache_fork::get_padding(/* flash_attn = */ false);
        const uint32_t k_rows = llama_kv_layer_rows(hparams, il, kv_size, swa_compress, n_ubatch, pad);
        size_t size = ggml_row_size(type_k, hparams.n_lora_kv + hparams.n_rot) * k_rows;
        if (hparams.indexer_head_size > 0 && hparams.n_swa > 0 &&
            il < (int) hparams.n_layer - (int) hparams.nextn_predict_layers &&
            !hparams.is_swa_impl[il]) {
            size += ggml_row_size(idx_type_k, hparams.indexer_head_size) * kv_size;
        }
        return size;
    }
    bool is_mla_attn = is_mla_model();
    if (is_mla_attn && mla_attn) {
        auto n_embd_head_qk_rope = hparams.n_rot;
        auto kv_lora_rank = hparams.n_lora_kv;
        if (flash_attn) {
            return ggml_row_size(type_k, kv_lora_rank + n_embd_head_qk_rope) * kv_size;
        }
        auto kv_type = mla_attn == 1 ? type_k : type_v;
        auto size = ggml_row_size(kv_type, kv_lora_rank + n_embd_head_qk_rope) * kv_size;
        if (mla_attn == 1) {
            size += ggml_row_size(type_v, kv_lora_rank*kv_size);
        }
        return size;
    }
    if (llm_arch_is_dsv4(arch)) {
        // mirrors llama_context::ensure_dsv4_cache_tensors
        constexpr uint32_t cache_pad = 256;
        const uint32_t csa_ratio = hparams.dsv4_csa_ratio;
        const uint32_t hca_ratio = hparams.dsv4_hca_ratio;

        const uint32_t n_stream = std::max<uint32_t>(1, n_seq_max);
        const uint32_t csa_kv = GGML_PAD(std::max<uint32_t>(1, (kv_size + csa_ratio - 1)/csa_ratio), cache_pad);
        const uint32_t hca_kv = GGML_PAD(std::max<uint32_t>(1, (kv_size + hca_ratio - 1)/hca_ratio), cache_pad);
        const uint32_t ratio = hparams.dsv4_compress_ratios[(size_t) il];
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_indexer_head = hparams.indexer_head_size;

        const uint32_t raw_pad = llama_kv_cache_fork::get_padding(flash_attn);
        const uint32_t k_rows = llama_kv_layer_rows(hparams, il, kv_size, swa_compress, n_ubatch, raw_pad);
        size_t size = ggml_row_size(type_k, n_embd_head) * hparams.n_head_kv(il) * k_rows;
        if (hparams.dsv4_shared_streams) {
            // readers alias their source, so only sources and key owners cost anything here
            const uint32_t n_rows = ratio == csa_ratio ? csa_kv : hca_kv;
            if (hparams.dsv41_is_kv_source(il)) {
                size += ggml_row_size(type_k, n_embd_head) * n_rows * n_stream;
                size += (size_t) n_embd_head * ratio * n_stream * sizeof(float) * 2;
            }
            if (hparams.dsv41_owns_index_k(il)) {
                size += ggml_row_size(idx_type_k, n_indexer_head) * n_rows * n_stream;
            }
            return size;
        }
        if (ratio == csa_ratio) {
            size += ggml_row_size(type_k, n_embd_head) * csa_kv * n_stream;
            size += ggml_row_size(idx_type_k, n_indexer_head) * csa_kv * n_stream;
            size += (size_t) 2 * n_embd_head * 2 * csa_ratio * n_stream * sizeof(float) * 2;
            size += (size_t) 2 * n_indexer_head * 2 * csa_ratio * n_stream * sizeof(float) * 2;
        } else if (ratio == hca_ratio) {
            size += ggml_row_size(type_k, n_embd_head) * hca_kv * n_stream;
            size += (size_t) n_embd_head * hca_ratio * n_stream * sizeof(float) * 2;
        }
        return size;
    }

    auto n_head_kv = hparams.n_head_kv(il);
    const uint32_t rows = llama_kv_layer_rows(hparams, il, kv_size, swa_compress && supports_swa_compress(), n_ubatch,
                                              llama_kv_cache_fork::get_padding(flash_attn));
    auto k_size = ggml_row_size(type_k, hparams.n_embd_head_k(il)) * n_head_kv*rows;
    auto v_size = ggml_row_size(type_v, hparams.n_embd_v_gqa(il)) * rows;
    // a qwen4exp sparse-attention layer also caches one raw indexer key per cell and one
    // pooled key per block of `ratio` cells
    if (arch == LLM_ARCH_QWEN4EXP && hparams.indexer_head_size > 0 && hparams.is_qsa(il)) {
        const uint32_t ratio = hparams.dsv4_compress_ratios[il];
        k_size += ggml_row_size(idx_type_k, hparams.indexer_head_size) * (rows + (rows + ratio - 1)/ratio);
    }
    // a PLE layer that is not recurrent still gets a state row for its convolution history
    // (Inkling's packed short-conv histories are sized the same way)
    if (hparams.n_embd_attn_layer_state(il) > 0) {
        auto state_sots = std::min<uint32_t>(std::max<uint32_t>(1, n_seq_max), kv_size);
        k_size += hparams.n_embd_attn_layer_state(il) * (size_t) state_sots * sizeof(float);
    }
    return k_size + v_size;
}

// S69-WIRE66-CONTEXT-MEMORY-ARCH-DECISION step 1: real accessor, not a stub -- wraps the
// `buft_layer` mapping the loader already populates per layer (src/llama.cpp model-load path,
// e.g. `model.buft_layer[il] = llama_default_buffer_type_offload(model, model.devices[...])`).
// Bounds-checked (unlike a raw `buft_layer[il]`) because this accessor is now called from layer
// loops keyed on `hparams.n_layer_eff()` (llama-kv-cache-dsv4.cpp, llama-memory-recurrent.cpp,
// llama-kv-cache.cpp), which is not guaranteed to equal `buft_layer.size()` for archs with nextn
// layers -- an out-of-range read here returns nullptr instead of undefined behavior.
// ADDED 2026-09-09, slice 25. Ported from 279b4bfaf^2:src/llama-model.cpp:1849-1866, with
// upstream's no_alloc branch deliberately NOT carried over. Upstream iterates pimpl->ctxs_bufs,
// which pairs each ggml_context with the buffers created from it, and under hparams.no_alloc
// recomputes each size with ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft) because its
// loader leaves a single placeholder buffer with a NULL base whose reported size is meaningless.
//
// Neither half of that applies here, and both were measured rather than assumed:
//   * NO PAIRING EXISTS. This fork keeps `ctxs` and `bufs` as separate, unpaired vectors on
//     llama_model. The loader builds one context per buffer type into a LOCAL ctx_map
//     (src/llama-load-tensors.cpp:224-234 and :385-394) which is never stored on the model, so
//     there is no (ctx, buft) pair to hand the size helper. The helper itself IS present
//     (ggml/include/ggml-alloc.h:80) -- the pairing is what is missing.
//   * IT IS NOT NEEDED. This fork's model-buffer allocation path has NO hparams.no_alloc branch:
//     measured over src/llama.cpp:4780-4990, 0 occurrences of no_alloc against a positive control
//     of 5 occurrences of model.bufs in the same range. Every buffer reaching model.bufs is a real
//     one (ggml_backend_cpu_buffer_from_ptr, ggml_backend_metal_buffer_from_ptr, or
//     ggml_backend_alloc_ctx_tensors_from_buft), so get_size is directly meaningful in every mode.
//     Porting upstream's branch would add an unreachable path behind an assert this fork's buffers
//     would FAIL, since their base is not null.
//
// IF THE FORK EVER GAINS A no_alloc ALLOCATION PATH, this function has to change with it -- that is
// the one condition under which the numbers below would start under-reporting.
//
// IT DID, 2026-09-27, and the second bullet above stopped being true. The absence of that path was
// itself the defect: common/fit.cpp sets no_alloc for its probes, the loader ignored it, and every
// probe was a real full load (row S69-FIT-NGL-AUTO-OVERCOMMITS-VRAM-ON-HYBRID-27B-F16-2026-09-27-01).
// The loader (llm_load_tensors, src/llama.cpp) now gives each weight context a zero-size dummy under
// no_alloc and records the projected bytes per buffer type in `no_alloc_sizes` AT LOAD TIME, while
// the (ctx, buft) pair is still in hand -- which answers the "no pairing exists" bullet without
// storing the loader's local ctx_map. Dummies report size 0, so the sum below is exact in both modes.
std::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (ggml_backend_buffer_t buf : bufs) {
        if (buf == nullptr) {
            continue;   // a null slot contributes nothing; skipping it is not a silent failure,
                        // there is no size to attribute and no buffer type to attribute it to
        }
        ret[ggml_backend_buffer_get_type(buf)] += ggml_backend_buffer_get_size(buf);
    }
    for (const auto & [buft, size] : no_alloc_sizes) {
        ret[buft] += size;
    }
    return ret;
}

// ADDED 2026-09-09, slice 30. The three per-layer RoPE accessors upstream declares and this fork
// never had. Every line below is transcribed from logic already live in this tree; nothing here is
// upstream code taken on faith, and nothing is a stub.
//
// SOURCES, so the transcription is checkable:
//   freq_base    src/llama-build-context.cpp:3279 and :3281-3283
//   freq_scale   src/llama-build-context.cpp:3280, with its non-SWA source CORRECTED to cparams --
//                see the note below, and the row filed against that line
//   factors      src/llama-build-context.cpp:527-539, transcribed field for field

float llama_model::get_rope_freq_base(const llama_cparams & cparams, int il) const {
    // The per-layer override wins outright, exactly as it does at build_std_attention:3281-3283
    // where it is applied AFTER the SWA choice and overwrites it unconditionally. This table is a
    // FORK capability with no upstream equivalent: porting upstream's accessor verbatim would drop
    // it silently for every architecture that sets it.
    if (hparams.has_rope_freq_base_per_layer) {
        return hparams.rope_freq_base_per_layer[il];
    }

    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    // NON-SWA READS CPARAMS, NOT hparams.rope_freq_scale_train. build_std_attention:3280 reads
    // hparams here while its own adjacent freq_base line (:3279) reads cparams, and three
    // independent sources say cparams is the correct one:
    //   src/llama.cpp:8831  cparams.rope_freq_scale IS hparams.rope_freq_scale_train unless the
    //                       user passed --rope-freq-scale, so reading hparams drops the override;
    //   src/llama.cpp:8868  forces cparams.rope_freq_scale = 1.0f when the scaling type is NONE,
    //                       which reading hparams also ignores;
    //   src/graphs/build_gemma4.cpp:191 reads cparams for this exact quantity, as does upstream.
    // That line in build_std_attention is filed as its own defect rather than changed from here --
    // it is a built, live path and deserves its own commit and its own controls.
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    // Transcribed from llm_build_context::build_rope_factors (src/llama-build-context.cpp:527-539),
    // including its unguarded layers[il] indexing. NO bounds check is added: the fork's helper has
    // none, and two helpers over one fact that disagree at the edge is a worse defect than either
    // choice -- the same reasoning src/llama-hparams.h:836-841 records for the is_swa accessors.

    // choose long/short freq factors based on the context size
    const uint32_t n_ctx_per_seq = cparams.n_ctx / cparams.n_seq_max;

    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_per_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

std::string llama_model::arch_name() const {
    // The fork's own spelling of upstream's llm_arch_name (src/llama-arch.h:843). Returns
    // std::string, matching upstream's signature, because the call site concatenates it.
    return llama_model_arch_name(arch);
}

size_t llama_model::n_tensors() const {
    // tensors_by_name's "for quantize-stats only" comment reads like a PARTIAL set, and this value
    // sizes graphs, so the population was checked rather than assumed. All five writes to the
    // member live in src/llama.cpp; the dominant one (4984-4989, under "// populate
    // tensors_by_name") loops every context and every tensor in the main load path with no
    // condition, and the other four append computed MLA tensors alongside it. No write is gated on
    // a quantize-stats path.
    // LIMIT: that is a STATIC read of the load path, not a runtime count from a loaded model. It
    // establishes that nothing restricts the population; it does not measure one.
    return tensors_by_name.size();
}

// The device a layer (or the output) RUNS on, from the buffer type the loader gave it. A HOST buffer type means the CPU device,
// whatever device the buffer type names: in a CUDA build the CPU layers get the pinned host type
// (llama_default_buffer_type_cpu(true) -> ggml_backend_cuda_host_buffer_type, src/llama.cpp:320-327), and that type's .device is
// CUDA0 (ggml-cuda.cu:1392). Returning it made every CPU layer read as CUDA0, so the upstream-ported memory classes, which place
// K/V and recurrent state on dev_layer(il) (llama_memory_hybrid_idx, llama_memory_recurrent, llama_kv_cache), put a CPU layer's
// cache in VRAM and the scheduler then ran its ops on the GPU. MEASURED 2026-11-10, gate binary fork-cuda-static, tiny glm5-next,
// n_gpu_layers 0: "offloaded 0/6 layers" yet "CUDA0 KV buffer size = 0.12 MiB" + "CUDA0 RS buffer size = 0.01 MiB", 279 CUDA0
// lines in the scheduler debug log, logits moved by up to 0.128 x rms vs CUDA_VISIBLE_DEVICES=-1 (the fork's own cache path,
// which reads buft_layer directly, stayed bit-identical). Upstream's dev_layer returns the CPU device for CPU layers
// (pimpl->dev_layer[il].dev), which is what this now matches.
static ggml_backend_dev_t llama_dev_of_buft(ggml_backend_buffer_type_t buft) {
    if (!buft) {
        return nullptr;
    }
    if (ggml_backend_buft_is_host(buft)) {
        if (ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            return cpu;
        }
    }
    return ggml_backend_buft_get_device(buft);
}

ggml_backend_dev_t llama_model::dev_output() const {
    // Mirrors dev_layer() below, against buft_output instead of buft_layer[il], including its
    // null-guard contract: an absent buft yields nullptr rather than a dereference. Upstream's
    // version returns pimpl->dev_output.dev -- a data layout this fork does not have -- so the
    // signature is upstream's and the body is this fork's.
    return llama_dev_of_buft(buft_output.buft);
}

ggml_backend_dev_t llama_model::dev_layer(uint32_t il) const {
    if (il >= buft_layer.size()) {
        return nullptr;
    }
    return llama_dev_of_buft(buft_layer[il].buft);
}


// PORTED from upstream ggml-org/llama.cpp master, src/llama-model.cpp
// (https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp, 153,165 chars,
// fetched raw and read in full), where it is llama_model::create_memory -- 23,707 chars, 487
// lines. Inserted from the extracted upstream text itself rather than retyped, so it cannot drift.
//
// MEASURED PORTABLE BEFORE IT WAS ATTEMPTED: the body references `pimpl` ZERO times, so it does not
// depend on upstream's pimpl-bearing llama_model; all 22 hparams/cparams fields it reads exist in
// our headers; and 9 of the 10 memory types it constructs were already declared here. The tenth,
// llama_memory_hybrid_idx, was a merge gap rather than a design difference and was fetched from
// upstream alongside this port.
llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    llama_memory_i * res;

    // the non-causal LFM2 decision graph reads the whole prompt in one batch, nothing is kept
    if (arch == LLM_ARCH_LFM2 && !hparams.causal_attn && hparams.n_layer_decision > 0) {
        return nullptr;
    }

    switch (arch) {
        // Models that need specific instantiation should be handled in the
        // switch statement
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_GEMMA_EMBEDDING2:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
        // DiffusionGemma is memoryless (the canvas is re-denoised from scratch; the prompt-KV store is the
        // model's own dg_pkv_* tensors, not a llama_memory). Without this case it fell to the default arm and
        // GGML_ASSERT(!hparams.is_swa_any()) aborted every llama_init_from_model since llama-context.cpp:1114
        // began calling create_memory() unconditionally (2026-09-29 metadata-only memory module).
        // Upstream lists it here too: unslothai/llama.cpp src/llama-model.cpp (diffusiongemma-24423-b10630).
        case LLM_ARCH_DIFFUSION_GEMMA:
        case LLM_ARCH_CLEF:
            {
                res = nullptr;
            } break;
        // LLM_ARCH_INKLING is NOT listed here any more (S69 P1): it takes the default arm, which builds the hybrid-iSWA
        // bookkeeping module (swa_type is STANDARD, set by its hparams loader). The module is metadata-only on this
        // graph path; the K/V cells and the per-layer conv-history row live in the legacy cache (llama_kv_cache_init).
        case LLM_ARCH_MINIMAX_M3:
            {
                // sparse (MSA) layers carry an indexer key cache, but leading dense layers do not
                llama_kv_cache::layer_filter_cb filter_idx =
                    [&](int32_t il) { return (uint32_t) il >= hparams.n_layer_dense_lead; };

                res = new llama_kv_cache_msa(
                        *this,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        1,
                        hparams.n_swa,
                        hparams.swa_type,
                        nullptr,
                        filter_idx,
                        nullptr,
                        params.metadata_only);
            } break;
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_DEEPSEEK32:
            {
                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && hparams.n_layer_nextn > 0) {
                    // The NextN/MTP draft head runs dense MLA (no DSA indexer), so the
                    // MTP context uses a plain attention KV cache holding only the
                    // nextn layer(s) - same pattern as the hybrid Qwen3.5 MTP context.
                    llama_kv_cache::layer_filter_cb filter =
                        [&](uint32_t il) { return il >= hparams.n_layer_eff(); };

                    res = new llama_kv_cache(
                            *this,
                            hparams,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            filter,
                            nullptr,
                            nullptr,
                            "",
                            params.metadata_only);
                } else {
                    // Main context: DSA cache for the trunk layers only - the nextn
                    // layer(s) are never attended by the trunk graph.
                    llama_kv_cache::layer_filter_cb filter_mla = nullptr;
                    if (hparams.n_layer_nextn > 0) {
                        filter_mla = [&](uint32_t il) { return il < hparams.n_layer_eff(); };
                    }
                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) { return il < hparams.n_layer_eff() && (arch != LLM_ARCH_GLM_DSA || hparams.is_indexer_full(il)); };

                    res = new llama_kv_cache_dsa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            filter_mla,
                            filter_lid,
                            nullptr,
                            params.metadata_only);
                }
            } break;
        case LLM_ARCH_HY_V4:
            {
                if (hparams.indexer_top_k == 0) {
                    // full-attention checkpoint: no indexer, so no indexer key cache
                    res = new llama_kv_cache(
                            *this,
                            hparams,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            "",
                            params.metadata_only);
                } else {
                    // only "full" layers own an indexer, so the shared layers need no indexer cache
                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) { return hparams.is_indexer_full(il); };

                    res = new llama_kv_cache_dsa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            filter_lid,
                            nullptr,
                            params.metadata_only);
                }
            } break;
        case LLM_ARCH_GLM5_NEXT:
            {
                // KDA layers are recurrent, the DSA layers use a K-only MLA cache plus an indexer cache.
                // tThe Nextn block is never attended by the trunk graph
                llama_memory_hybrid_idx::layer_filter_cb filter_attn = [&](uint32_t il) {
                    return il < hparams.n_layer_eff() && !hparams.is_recr(il);
                };
                llama_memory_hybrid_idx::layer_filter_cb filter_idx = [&](uint32_t il) {
                    return il < hparams.n_layer_eff() && !hparams.is_recr(il) && hparams.is_indexer_full(il);
                };
                llama_memory_hybrid_idx::layer_filter_cb filter_recr = [&](uint32_t il) {
                    return il < hparams.n_layer_eff() && hparams.is_recr(il);
                };

                // the draft head is a single DSA layer
                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                    if (hparams.n_layer_nextn == 0) {
                        throw std::runtime_error("GLM5-Next MTP requires the NextN block, convert without --no-mtp");
                    }
                    filter_attn = [&](uint32_t il) { return il >= hparams.n_layer_eff(); };
                    filter_idx  = [&](uint32_t il) { return il >= hparams.n_layer_eff(); };
                    filter_recr = [&](uint32_t)    { return false; };
                }

                res = new llama_memory_hybrid_idx(
                    /* model             */ *this,
                    /* attn_type_k       */ params.type_k,
                    /* attn_type_v       */ params.type_v,
                    /* attn_v_trans      */ !cparams.flash_attn,
                    /* attn_kv_size      */ cparams.n_ctx_seq,
                    /* attn_n_pad        */ 1,
                    /* attn_n_swa        */ hparams.n_swa,
                    /* attn_swa_type     */ hparams.swa_type,
                    /* recurrent_type_r  */ GGML_TYPE_F32,
                    /* recurrent_type_s  */ GGML_TYPE_F32,
                    /* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                    /* n_seq_max         */ cparams.n_seq_max,
                    /* n_rs_seq          */ cparams.n_rs_seq,
                    /* offload           */ cparams.offload_kqv,
                    /* unified           */ cparams.kv_unified,
                    /* filter_attn       */ std::move(filter_attn),
                    /* filter_recr       */ std::move(filter_recr),
                    /* filter_idx        */ std::move(filter_idx));
            } break;
        case LLM_ARCH_DOTS3NOTE:
            {
                GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && hparams.n_layer_nextn > 0) {
                    // MTP draft context: plain attention KV cache holding only the nextn layer
                    llama_kv_cache::layer_filter_cb filter =
                        [&](uint32_t il) { return il >= hparams.n_layer_eff(); };

                    res = new llama_kv_cache(
                            *this,
                            hparams,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            filter,
                            nullptr,
                            nullptr,
                            "",
                            params.metadata_only);
                } else {
                    // main context: DSA cache for the trunk full-attention layers plus a window-sized SWA cache
                    llama_kv_cache::layer_filter_cb filter_mla = nullptr;
                    if (hparams.n_layer_nextn > 0) {
                        filter_mla = [&](uint32_t il) { return il < hparams.n_layer_eff(); };
                    }
                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) { return il < hparams.n_layer_eff() && hparams.is_indexer_full(il); };

                    res = new llama_kv_cache_dsa_iswa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            1,
                            filter_mla,
                            filter_lid,
                            nullptr,
                            params.metadata_only);
                }
            } break;
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:   // ik_llama #2455: same memory layout as deepseek4
            {
                GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                    const llama_memory_i::layer_filter_cb filter_mtp = [&](int32_t il) {
                        return il >= (int32_t) hparams.n_layer_eff();
                    };

                    res = new llama_kv_cache_iswa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            1,
                            nullptr,
                            filter_mtp,
                            nullptr,
                            nullptr,
                            params.metadata_only);
                } else {
                    res = new llama_kv_cache_dsv4(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            1,
                            cparams.n_rs_seq,
                            nullptr,
                            nullptr,
                            params.metadata_only);
                }
            } break;
        case LLM_ARCH_DFLASH:
            {
                // DSV4 DSpark stages store a single MLA-style K per position (window = the draft ring)
                if (hparams.dsv4_hc_mult > 0) {
                    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                    res = new llama_kv_cache_iswa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            1,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            params.metadata_only);
                    break;
                }
            }
            [[fallthrough]];
        // Models that need standard caching should rely on recurrent/hybrid
        // checks
        default:
            {
                // Dense MTP heads use a plain attention KV cache instead of the hybrid wrapper.
                const bool mtp_on_hybrid_qwen =
                    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
                    (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ||
                     arch == LLM_ARCH_BAILINGMOE3);

                const bool mtp_on_hybrid_nemotron =
                    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && arch == LLM_ARCH_NEMOTRON_H_MOE;

                if (llm_arch_is_recurrent(arch)) {
                    res = new llama_memory_recurrent(
                            *this,
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            cparams.offload_kqv,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max,
                            cparams.n_rs_seq,
                            nullptr,
                            params.metadata_only);
                } else if (llm_arch_is_hybrid(arch) && !mtp_on_hybrid_qwen && !mtp_on_hybrid_nemotron) {
                    // The main difference between hybrid architectures is the
                    // layer filters, so pick the right one here
                    llama_memory_hybrid::layer_filter_cb filter_attn = nullptr;
                    llama_memory_hybrid::layer_filter_cb filter_recr = nullptr;
                    // only the sparse-attention architectures use llama_memory_hybrid_idx
                    // a null filter_idx means the GGUF has no indexer tensors
                    llama_memory_hybrid::layer_filter_cb filter_idx  = nullptr;
                    const bool needs_mem_idx = (arch == LLM_ARCH_QWEN4EXP);
                    if (arch == LLM_ARCH_FALCON_H1) {
                        filter_attn = [&](uint32_t) { return true; };
                        filter_recr = [&](uint32_t) { return true; };
                    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
                        filter_attn = [&](uint32_t il) {
                            return !hparams.is_recr(il) && hparams.n_ff(il) == 0;
                        };
                        filter_recr = [&](uint32_t il) {
                            return hparams.is_recr(il) && hparams.n_ff(il) == 0;
                        };
                    } else if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_MINIMAX_01) {
                        filter_attn = [&](uint32_t il) {
                            return il < hparams.n_layer_eff() && !hparams.is_recr(il);
                        };
                        filter_recr = [&](uint32_t il) {
                            return il < hparams.n_layer_eff() && hparams.is_recr(il);
                        };

                        if (arch == LLM_ARCH_QWEN4EXP && hparams.indexer_head_size > 0) {
                            // QSA runs on the dense-attention layers only
                            filter_idx = [&](uint32_t il) {
                                return il < hparams.n_layer_eff() && !hparams.is_recr(il);
                            };
                        }
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        // Use hybrid-iswa for hybrid models with SWA
                        res = new llama_memory_hybrid_iswa(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_swa_full     */ params.swa_full,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_ubatch     */ cparams.n_ubatch,
                            /* attn_n_pad        */ 1,
                            /* recurrent_type_r  */ GGML_TYPE_F32,
                            /* recurrent_type_s  */ GGML_TYPE_F32,
                            /* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr),
                            /* metadata_only     */ params.metadata_only);
                    } else if (needs_mem_idx) {
                        // sparse attention over a per-token indexer cache, in its own memory type
                        res = new llama_memory_hybrid_idx(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_pad        */ 1,
                            /* attn_n_swa        */ hparams.n_swa,
                            /* attn_swa_type     */ hparams.swa_type,
                            /* recurrent_type_k  */ GGML_TYPE_F32,
                            /* recurrent_type_v  */ GGML_TYPE_F32,
                            /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr),
                            /* filter_idx        */ std::move(filter_idx),
                            /* metadata_only     */ params.metadata_only);
                    } else {
                        res = new llama_memory_hybrid(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_pad        */ 1,
                            /* attn_n_swa        */ hparams.n_swa,
                            /* attn_swa_type     */ hparams.swa_type,
                            /* recurrent_type_k  */ GGML_TYPE_F32,
                            /* recurrent_type_v  */ GGML_TYPE_F32,
                            /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr),
                            /* metadata_only     */ params.metadata_only);
                    }
                } else {
                    llama_kv_cache::layer_filter_cb filter = nullptr;
                    llama_memory_i::layer_reuse_cb reuse = nullptr;
                    llama_kv_cache::layer_share_cb share = nullptr;

                    if (arch == LLM_ARCH_GEMMA3N || arch == LLM_ARCH_GEMMA4) {
                        reuse = [&](uint32_t il) {
                            GGML_ASSERT(hparams.n_layer_kv_from_start >= 2);

                            if (il >= (uint32_t)hparams.n_layer_kv_from_start) {
                                return hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
                            }

                            return -1;
                        };
                    }

                    if (mtp_on_hybrid_qwen || mtp_on_hybrid_nemotron) {
                        filter = [&](uint32_t il) { return il >= hparams.n_layer_eff(); };
                    }

                    // don't filter when n_layer_nextn is repurposed for a router layer the trunk attends
                    // or when a model is entirely n_layer_nextn layers and has no trunk
                    if (hparams.n_layer_nextn > 0 && hparams.n_layer_eff() > 0 && hparams.router_layer < 0) {
                        if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                            filter = [&](uint32_t il) { return il >= hparams.n_layer_eff(); };
                        } else {
                            filter = [&](uint32_t il) { return il <  hparams.n_layer_eff(); };
                        }
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        GGML_ASSERT(hparams.is_swa_any());

                        if (arch == LLM_ARCH_GEMMA4_ASSISTANT) {
                            llama_memory_t mem_other = llama_get_memory(cparams.ctx_other);

                            // The TOTAL layer count is correct here, despite the sibling arm ~20
                            // lines above using the MTP-excluding hparams.n_layer_kv_from_start for
                            // the same -2/-1 "last two layers" idiom. MEASURED 2026-09-09: none of
                            // the three GEMMA4* arches ever loads LLM_KV_NEXTN_PREDICT_LAYERS --
                            // src/models/gemma4.cpp and src/graphs/build_gemma4.cpp mention it zero
                            // times, and neither the LLM_ARCH_GEMMA4 case nor the shared
                            // GEMMA4_MTP/GEMMA4_ASSISTANT case in llm_load_hparams reads it
                            // (control: the LLM_ARCH_STEP35 case does). So nextn == 0 for any model
                            // that can be ctx_other here, and TOTAL == EFFECTIVE.
                            //
                            // If a Gemma-4 backbone ever DOES ship nextn layers, this must become
                            // llama_model_n_layer(model_other) - llama_model_n_layer_nextn(model_other)
                            // - (is_swa ? 2 : 1), or it will index into the MTP layers.
                            // Row: S69-GEMMA4-ASSISTANT-SHARE-USES-OTHER-MODELS-TOTAL-LAYER-COUNT-2026-09-09-01
                            share = [&](int32_t il) {
                                const llama_model * model_other = llama_get_model(cparams.ctx_other);

                                if (hparams.is_swa(il)) {
                                    return llama_model_n_layer(model_other) - 2;
                                }

                                return llama_model_n_layer(model_other) - 1;
                            };

                            res = new llama_kv_cache_iswa(
                                    *this,
                                    params.type_k,
                                    params.type_v,
                                    !cparams.flash_attn,
                                    cparams.offload_kqv,
                                    params.swa_full,
                                    cparams.kv_unified,
                                    cparams.n_ctx_seq,
                                    cparams.n_seq_max,
                                    cparams.n_ubatch,
                                    1,
                                    mem_other,
                                    filter,
                                    reuse,
                                    share,
                                    params.metadata_only);
                        } else {
                            res = new llama_kv_cache_iswa(
                                    *this,
                                    params.type_k,
                                    params.type_v,
                                    !cparams.flash_attn,
                                    cparams.offload_kqv,
                                    params.swa_full,
                                    cparams.kv_unified,
                                    cparams.n_ctx_seq,
                                    cparams.n_seq_max,
                                    cparams.n_ubatch,
                                    1,
                                    nullptr,
                                    filter,
                                    reuse,
                                    share,
                                    params.metadata_only);
                        }
                    } else {
                        GGML_ASSERT(!hparams.is_swa_any());

                        res = new llama_kv_cache(
                                *this,
                                hparams,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                1,
                                hparams.n_swa,
                                hparams.swa_type,
                                nullptr,
                                filter,
                                nullptr,
                                nullptr,
                                "",
                                params.metadata_only);
                    }
                }
            }
    }

    return res;
}

const llama_hadamard_transform * llama_model::hadamard_rotation(const struct ggml_tensor * t) const {
    // qwen4exp (Gyro) uses the same prism.hadamard kvs as qwen35 (Prism)
    if (!t || (arch != LLM_ARCH_QWEN35 && arch != LLM_ARCH_QWEN4EXP)) {
        return nullptr;
    }
    const auto it = hadamard_map.find(t->name);
    return it != hadamard_map.end() && it->second.block_size > 0 ? &it->second : nullptr;
}

// Make model.hdmd.rot / model.hdmd.inv, the data the class-path graphs read (llm_graph_context::build_hadamard_input,
// llama_verify_hadamard_graph), from the names / sign data / flags the loader recorded in model.hdmd from the same GGUF metadata
// that fills the fork's hadamard_map. Layout and construction follow upstream's load_tensors_hadamard (ggml-org/llama.cpp
// src/llama-model.cpp, MIT, read in full): rot[weight] = { block x block F32 matrix H[r][c] = (-1)^popcount(r&c)/sqrt(block),
// sign vector or nullptr, GDN value-head permutation }, one matrix per (block size, buffer type) and one sign tensor per
// (width, buffer type), both on the buffer type of the weight they serve (CPU extra types map to the plain CPU type), and the
// inverse transforms on the buffer type of the forward ones. Row S69-ENGREV-UPSTREAM-MERGE-PRISM-HADAMARD-TWO-IMPLEMENTATIONS-2026-11-04-01.
size_t llama_hdmd_fill(llama_model & model) {
    llama_hadamard & hd = model.hdmd;
    if (hd.weight_blocks.empty() && hd.inverse_blocks.empty()) {
        return 0;
    }
    hd.rot.clear();
    hd.inv.clear();

    std::unordered_map<std::string, ggml_tensor *> by_name;
    for (const auto & [name, t] : model.tensors_by_name) {
        by_name.emplace(name, t);
    }

    struct rotation_entry {
        uint32_t block_size;
        ggml_backend_buffer_type_t buft;
        ggml_tensor * tensor;
    };
    std::vector<rotation_entry> rotations;
    std::map<std::pair<uint32_t, ggml_backend_buffer_type_t>, ggml_tensor *> sign_tensors;

    // a tied output head reads the token embedding table as its weight (upstream: weight_blocks gets token_embd.weight)
    if (hd.tied_output) {
        const auto it = hd.inverse_blocks.find("token_embd.weight");
        if (it == hd.inverse_blocks.end()) {
            throw std::runtime_error("prism.hadamard.tied_output requires a latent token embedding");
        }
        hd.weight_blocks.emplace("token_embd.weight", it->second);
    }

    const std::pair<const std::unordered_map<std::string, uint32_t> *, llama_hadamard_rotations *> groups[] = {
        { &hd.weight_blocks,  &hd.rot },
        { &hd.inverse_blocks, &hd.inv },
    };
    ggml_backend_buffer_type_t preferred_buft = nullptr;

    // one context + buffer owned by the model for each new tensor, like the fork's own sign tensors
    const auto make_tensor = [&](ggml_backend_buffer_type_t buft, ggml_type type, int64_t ne0, int64_t ne1, const char * name,
                                 const std::vector<float> & data) -> ggml_tensor * {
        ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create Hadamard tensor context");
        }
        ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, type, ne0, ne1) : ggml_new_tensor_1d(ctx, type, ne0);
        ggml_set_name(t, name);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf == nullptr) {
            ggml_free(ctx);
            throw std::runtime_error(format("unable to allocate %s Hadamard buffer", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
        model.ctxs.push_back(ctx);
        model.bufs.push_back(buf);
        return t;
    };

    for (const auto & [blocks, target] : groups) {
        for (const auto & entry : *blocks) {
            const std::string & weight_name = entry.first;
            const uint32_t block_size = entry.second;

            ggml_tensor * weight = nullptr;
            if (hd.tied_output && weight_name == "token_embd.weight") {
                weight = target == &hd.rot ? model.output : model.tok_embd;
                if (!weight || strcmp(weight->name, "token_embd.weight") != 0) {
                    throw std::runtime_error("prism.hadamard.tied_output is not bound to the token embedding");
                }
            } else if (const auto it = by_name.find(weight_name); it != by_name.end()) {
                weight = it->second;
            }
            if (weight == nullptr) {
                throw std::runtime_error(format("prism.hadamard weight not found: %s", weight_name.c_str()));
            }
            if (block_size == 0 || weight->ne[0] % block_size != 0) {
                throw std::runtime_error(format("prism.hadamard block size %u does not divide input dimension %lld for %s",
                        block_size, (long long) weight->ne[0], weight_name.c_str()));
            }

            // no buffer = a fit probe / no_alloc load: the transform goes on the CPU buffer type
            ggml_backend_buffer_type_t buft = weight->buffer ? ggml_backend_buffer_get_type(weight->buffer)
                                                             : ggml_backend_cpu_buffer_type();
            // CPU extra buffer types (e.g. repack) only accept tensors they can repack
            if (ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft)) {
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    buft = ggml_backend_dev_buffer_type(dev);
                }
            }
            if (target == &hd.rot) {
                preferred_buft = buft;
            } else if (preferred_buft) {
                buft = preferred_buft;
            }

            auto it = std::find_if(rotations.begin(), rotations.end(), [&](const rotation_entry & r) {
                return r.block_size == block_size && r.buft == buft;
            });
            if (it == rotations.end()) {
                std::vector<float> data((size_t) block_size * block_size);
                const float scale = 1.0f / sqrtf((float) block_size);
                for (uint32_t row = 0; row < block_size; ++row) {
                    for (uint32_t col = 0; col < block_size; ++col) {
                        uint32_t parity = row & col;
                        parity ^= parity >> 16;
                        parity ^= parity >> 8;
                        parity ^= parity >> 4;
                        parity ^= parity >> 2;
                        parity ^= parity >> 1;
                        data[(size_t) row * block_size + col] = (parity & 1) ? -scale : scale;
                    }
                }
                char rotation_name[GGML_MAX_NAME];
                snprintf(rotation_name, sizeof(rotation_name), "prism.hadamard.%u", block_size);
                rotations.push_back({ block_size, buft, make_tensor(buft, GGML_TYPE_F32, block_size, block_size, rotation_name, data) });
                it = std::prev(rotations.end());
            }

            ggml_tensor * sign_tensor = nullptr;
            if (!hd.sign_data.empty()) {
                const uint32_t width = (uint32_t) weight->ne[0];
                const auto sd = hd.sign_data.find(width);
                if (sd == hd.sign_data.end()) {
                    throw std::runtime_error(format("prism.hadamard has no sign vector for width %u (%s)", width, weight_name.c_str()));
                }
                const auto key = std::make_pair(width, buft);
                auto st = sign_tensors.find(key);
                if (st == sign_tensors.end()) {
                    std::vector<float> data(width);
                    for (uint32_t i = 0; i < width; ++i) {
                        data[i] = (float) sd->second[i];
                    }
                    char sign_name[GGML_MAX_NAME];
                    snprintf(sign_name, sizeof(sign_name), "prism.hadamard.signs.%u", width);
                    st = sign_tensors.emplace(key, make_tensor(buft, GGML_TYPE_F32, width, 0, sign_name, data)).first;
                }
                sign_tensor = st->second;
            }

            llama_hdmd_transform transform { it->tensor, sign_tensor };
            // the GDN output projection reads its value heads in tiled order, see set_gdn_v_perm
            if (hd.gdn_v_grouped && weight_name.find(".ssm_out.") != std::string::npos &&
                !transform.set_gdn_v_perm(weight->ne[0], model.hparams.ssm_dt_rank, model.hparams.ssm_n_group)) {
                throw std::runtime_error(format("prism.hadamard: bad GDN head geometry for %s", weight_name.c_str()));
            }
            target->emplace(weight, transform);
        }
    }

    LLAMA_LOG_INFO("%s: hdmd filled: %zu Hadamard-folded weight(s) (%zu inverse-lookup), %zu rotation(s), %zu sign vector(s)\n",
            __func__, hd.rot.size() + hd.inv.size(), hd.inv.size(), rotations.size(), sign_tensors.size());
    return hd.rot.size() + hd.inv.size();
}

// upstream helpers for src/models classes, on the fork base (upstream bodies read hparams.n_layer() and ml.load_mtp;
// the fork spells them n_layer_eff() and the model's mtp flag, and reads SWA patterns with llm_load_swa_pattern)
llama_model_base::nextn_flags_t llama_model_base::nextn_flags(llama_model_loader & ml, llm_tensor trunk_probe) const {
    int trunk_f = 0;
    int mtp_f   = 0;

    // a file without the first trunk layer is MTP-only, a file without the first NextN layer is trunk-only
    if (hparams.n_layer_nextn > 0) {
        if (ml.get_weight(tn(trunk_probe, "weight", 0).str().c_str()) == nullptr) {
            trunk_f = TENSOR_NOT_REQUIRED;
        }
        if (ml.get_weight(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", hparams.n_layer_eff()).str().c_str()) == nullptr) {
            mtp_f = TENSOR_NOT_REQUIRED;
        }
    }

    if (!mtp) {
        mtp_f |= TENSOR_SKIP;
    }

    return { trunk_f, mtp_f };
}

void llama_model_base::load_swa_pattern(llama_model_loader & ml, uint32_t n_pattern, bool dense_first) {
    llm_load_swa_pattern(ml, hparams, n_pattern, dense_first);
}

uint32_t llama_model_n_tokens_delay(const struct llama_model * model) {
    return model->hparams.n_tokens_delay;
}

// upstream helper (merge 2026-11-04, MERGE_HEAD src/llama-model.cpp:3960)
// read rows [row0, row0 + n_rows) of a 2D tensor as F32
static void llama_tensor_rows_to_f32(const ggml_tensor * tensor, int64_t row0, int64_t n_rows, float * out) {
    const size_t   nelements = (size_t) n_rows * tensor->ne[0];
    const size_t   offset    = (size_t) row0 * tensor->nb[1];
    const size_t   nbytes    = ggml_row_size(tensor->type, tensor->ne[0]) * n_rows;

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, out, offset, nbytes);
        return;
    }

    std::vector<uint8_t> buf(nbytes);
    ggml_backend_tensor_get(tensor, buf.data(), offset, nbytes);

    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) buf.data(), out, nelements);
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) buf.data(), out, nelements);
    } else if (ggml_is_quantized(tensor->type) && traits->to_float != nullptr) {
        traits->to_float(buf.data(), out, nelements);
    } else {
        GGML_ABORT("unsupported tensor type for dequantization: %s", ggml_type_name(tensor->type));
    }
}

uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out) {
    if (model->vocab.n_tokens() == 0 || model->tok_embd == nullptr || tokens == nullptr || out == nullptr) {
        return 0;
    }

    const ggml_tensor * tensor = model->tok_embd;
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= tensor->ne[1]) {
            return 0;
        }
        llama_tensor_rows_to_f32(tensor, tokens[i], 1, out + (size_t) i * tensor->ne[0]);
    }

    return n_tokens * (uint32_t) tensor->ne[0];
}

// upstream (MERGE_HEAD src/llama-model.cpp:3844): per-model switch read by llama_kv_cache (llama-kv-cache.cpp, "see llama_model_set_attn_rot")
void llama_model_set_attn_rot(struct llama_model * model, bool value) {
    model->attn_rot = value;
}
