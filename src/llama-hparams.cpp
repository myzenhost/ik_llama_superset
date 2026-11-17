
#include "llama-hparams.h"
#include "llama-yarn.h"
#include "llama-model-loader.h"
#include "llama-model.h"
// llama-hparams.h forward-declares `enum llm_ffn_op_type : int;` only, which is enough to declare a
// member of that type and NOT enough to name its enumerators. LLM_FFN_OP_TYPES_FROM_STRING below
// needs the enumerators, so the definition has to be visible here.
#include "llama-graph.h"
#include <string>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>

#define LLAMA_MAX_EXPERTS 512  // Qwen3 Next

// LONGROPE ADDED 2026-09-09. Its absence was not cosmetic: llama_rope_scaling_type_from_string()
// below iterates THIS map and returns UNSPECIFIED for anything missing from it, and line ~207 then
// asserts the result is not UNSPECIFIED. So a GGUF declaring rope_scaling.type = "longrope"
// (Phi-3-128k, Phi-3.5) did not degrade -- it ABORTED on that assert and could not load at all.
// Meanwhile 16 sites already branch on LLAMA_ROPE_SCALING_TYPE_LONGROPE (src/llama-load-tensors.cpp
// twice, and apertus, deci, granite, granite-moe, granite-swa, llama, minicpm, mistral3,
// pangu-embed, refact, step35 x2), and include/llama.h:289-290 both defines the value and makes it
// LLAMA_ROPE_SCALING_TYPE_MAX_VALUE. Every one of those readers was unreachable: a producerless
// value with twelve architectures waiting on it. The four-entry table is upstream's, read in full
// at 279b4bfaf^2:src/llama-model.cpp:964-969.
static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,     "none"     },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,   "linear"   },
    { LLAMA_ROPE_SCALING_TYPE_YARN,     "yarn"     },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE, "longrope" },
};

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

const char * llama_hparams::rope_scaling_type_name(llama_rope_scaling_type type) {
    return LLAMA_ROPE_SCALING_TYPES.at(type);
}

// ADDED 2026-09-09. This free function was DECLARED at src/llama-model.h:39 and CALLED at
// src/llama-model-saver.cpp:320, and defined NOWHERE in the tree -- the declaration's own comment
// claimed "defined in llama-model.cpp", which was false. The merge took upstream's declaration and
// upstream's caller and left the definition behind in a file this fork splits differently.
//
// IT LIVES HERE, NOT IN llama-model.cpp WHERE THE COMMENT CLAIMED, because the table it reads is
// here. A second copy of the table in another translation unit is precisely the duplicate-symbol
// shape that stopped the built set linking in slice 33 (llama_n_ctx_train, llama_n_layer,
// llama_vocab_type). One table, one owner.
//
// Body adopted verbatim from 279b4bfaf^2:src/llama-model.cpp:971-973. It uses .at(), which throws
// on LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED -- upstream behaves identically, and the load path asserts
// that value never survives, so the throw is unreachable through the loader. Kept identical to
// upstream rather than "improved": a silent divergence in a round-tripped GGUF field is worse than
// a shared, documented edge.
std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type) {
    return LLAMA_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

// ADDED 2026-09-09, same shape as the one above: DECLARED at src/llama-model.h:40, CALLED at
// src/models/modern-bert.cpp:22, defined nowhere. It is only latent because our src/CMakeLists.txt
// does not currently consume ${LLAMA_MODELS_SOURCES}, so modern-bert.cpp is never compiled -- the
// moment models/ is wired, this becomes a link error rather than a dormant one.
//
// Table and body adopted from 279b4bfaf^2:src/llama-model.cpp:990-1005. Upstream's comment there
// records the intent: it maps the GGUF `<arch>.hidden_activation` string to the FFN op the graph
// builders use, lists only gated activations that map cleanly, and falls back to GeGLU because that
// is the historical default for ModernBert-style architectures -- which is why the caller passes
// LLM_FFN_GEGLU as its fallback. All four target enumerators were verified present in this fork's
// llm_ffn_op_type (src/llama-graph.h:53) before this table was written.
static const std::map<std::string, llm_ffn_op_type> LLM_FFN_OP_TYPES_FROM_STRING = {
    { "gelu",   LLM_FFN_GEGLU  },
    { "geglu",  LLM_FFN_GEGLU  },
    { "silu",   LLM_FFN_SWIGLU },
    { "swish",  LLM_FFN_SWIGLU },
    { "swiglu", LLM_FFN_SWIGLU },
    { "relu",   LLM_FFN_RELU   },
    { "reglu",  LLM_FFN_REGLU  },
};

llm_ffn_op_type llm_ffn_op_type_from_string(const std::string & name, llm_ffn_op_type fallback) {
    const auto it = LLM_FFN_OP_TYPES_FROM_STRING.find(name);
    if (it != LLM_FFN_OP_TYPES_FROM_STRING.end()) {
        return it->second;
    }
    return fallback;
}

static inline const char * llm_expert_gating_func_name(llm_expert_gating_func_type type) {
    switch (type) {
        case LLM_EXPERT_GATING_FUNC_SOFTMAX: return "softmax";
        case LLM_EXPERT_GATING_FUNC_SIGMOID: return "sigmoid";
        case LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT: return "weight";
        case LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS: return "sqrtsoftplus";
        default: return "none";
    }
}

// In-Place TTT keys (gguf-py Keys.TTT, written by conversion/qwen.py Qwen3Model._set_ttt_parameters). Absent chunk
// key = no TTT. Official defaults and meaning: ByteDance-Seed/In-Place-TTT be2324829b0e
// inference_model/hf_qwen3/configuration_qwen3.py:184-240. Row S69-FORK-INPLACE-TTT-2026-10-26-01.
static void load_ttt_params(llama_model_loader & ml, llama_hparams & hparams) {
    hparams.ttt_chunk = 0;
    std::fill(hparams.ttt_layer_impl.begin(), hparams.ttt_layer_impl.end(), false);
    uint32_t chunk = 0;
    if (!ml.get_key(LLM_KV_TTT_CHUNK, chunk, false)) {
        return;
    }
    if (chunk == 0) {
        throw std::runtime_error("ttt: chunk must be > 0");
    }
    std::vector<int32_t> layers;
    ml.get_arr(LLM_KV_TTT_LAYERS, layers, true);
    uint32_t n_on = 0;
    for (int32_t id : layers) {
        if (id < 0 || (uint32_t) id >= hparams.n_layer) {
            throw std::runtime_error(format("ttt: layer id %d outside [0, %u)", id, hparams.n_layer));
        }
        n_on += hparams.ttt_layer_impl[id] ? 0 : 1;
        hparams.ttt_layer_impl[id] = true;
    }
    ml.get_key(LLM_KV_TTT_LR, hparams.ttt_lr, true);
    ml.get_key(LLM_KV_TTT_TARGET,       hparams.ttt_target,       false);
    ml.get_key(LLM_KV_TTT_CONV_PADDING, hparams.ttt_conv_padding, false);
    ml.get_key(LLM_KV_TTT_CLIP_TAU,     hparams.ttt_clip_tau,     false);
    if (hparams.ttt_target > 1) {
        throw std::runtime_error(format("ttt: target %u unknown (0 = hidden_states, 1 = input_embed)", hparams.ttt_target));
    }
    if (hparams.ttt_conv_padding > 1) {
        throw std::runtime_error(format("ttt: conv_padding %u unknown (0 = symmetric, 1 = causal)", hparams.ttt_conv_padding));
    }
    if (!(hparams.ttt_clip_tau >= 0.0f)) {
        throw std::runtime_error("ttt: clip_tau must be >= 0");
    }
    hparams.ttt_chunk = chunk;
    LLAMA_LOG_INFO("%s: in-place TTT on %u layer(s): chunk %u, lr %g, target %s, conv %s, clip_tau %g\n", __func__,
            n_on, chunk, hparams.ttt_lr, hparams.ttt_target ? "input_embed" : "hidden_states",
            hparams.ttt_conv_padding ? "causal" : "symmetric", hparams.ttt_clip_tau);
}

// Qwen3-VL deepstack layer count.
// llama.cpp's own GGUFs store it as "<arch>.n_deepstack_layers". The GGUFs Ollama writes (our
// models/ollama store, e.g. the qwen3-vl:8b blob) do not: they list the vision layers that feed deepstack in
// "<arch>.vision.deepstack_visual_indexes" ([8, 16, 24] in that blob, read with gguf.GGUFReader 2026-10-05), and the
// count of that array IS the deepstack layer count (the unsloth mmproj for the same model carries 3 deepstack
// mergers and an output width of 4 x 4096 = 16384). Without the fallback the text model reads 0, its input width
// stays 4096 and mtmd refuses the projector ("mismatch between text model (n_embd = 4096) and mmproj (n_embd = 16384)",
// llama-server exit 1, row S69-VL-SERVE-ENDPOINT-UNSLOTH-MMPROJ-01). The native key wins when both exist.
static void load_qwen3vl_deepstack_layers(llama_model_loader & ml, llama_hparams & hparams) {
    if (ml.get_key(LLM_KV_NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers, false)) {
        return;
    }
    std::string key = ml.llm_kv(LLM_KV_NUM_DEEPSTACK_LAYERS);              // "<arch>.n_deepstack_layers"
    const std::string suffix = ".n_deepstack_layers";
    if (key.size() <= suffix.size() || key.compare(key.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return;
    }
    key.resize(key.size() - suffix.size());
    key += ".vision.deepstack_visual_indexes";
    uint32_t n = 0;
    if (ml.get_arr_n(key, n, false) && n > 0) {
        hparams.n_deepstack_layers = n;
        LLAMA_LOG_INFO("%s: n_deepstack_layers = %u taken from %s (no native n_deepstack_layers key)\n", __func__, n, key.c_str());
    }
}

// qwen3vl_pin_input_width (the comment the two call sites point at):
// this fork widens hparams.n_embd to (1 + n_deepstack_layers) * n_embd at load time (llama-hparams.cpp QWEN3VL / QWEN3VLMOE,
// undone by the "/ (n_deepstack_layers + 1)" in llama-load-tensors.cpp and build_qwen3.cpp). hparams.n_embd_inp() then added
// n_embd * n_deepstack_layers a SECOND time, so llama_model_n_embd_inp() answered 4x the true width for a 3-layer deepstack
// model (measured 2026-10-05 on a random-weight qwen3vl: native key 3 -> 4096, true input width 1024) and mtmd refused every
// projector. Pinning n_embd_inp_impl (the explicit-width field n_embd_inp() returns first, llama-hparams.h:748) to the
// widened n_embd makes the reported input width the one the graph really consumes.

static bool load_dflash_target_layer_ids(
        llama_model_loader & ml,
        const std::string & key,
        llama_hparams & hparams,
        bool required) {
    const int kid = gguf_find_key(ml.meta, key.c_str());
    if (kid < 0 || gguf_get_kv_type(ml.meta, kid) != GGUF_TYPE_ARRAY) {
        if (required) {
            throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
        }
        return false;
    }

    const enum gguf_type type = gguf_get_arr_type(ml.meta, kid);
    if (type != GGUF_TYPE_UINT32 && type != GGUF_TYPE_INT32) {
        throw std::runtime_error(format("dflash: %s must be a uint32/int32 array", key.c_str()));
    }

    uint32_t n = 0;
    ml.get_arr_n(key, n, true);
    if (n == 0) {
        throw std::runtime_error(format("dflash: %s must not be empty", key.c_str()));
    }
    if (n > 8) {
        throw std::runtime_error(format("dflash: %s has %u entries, max is 8", key.c_str(), n));
    }

    hparams.dflash_n_target_layers = n;
    for (uint32_t & id : hparams.dflash_target_layer_ids) {
        id = 0;
    }

    if (type == GGUF_TYPE_INT32) {
        std::array<int32_t, 8> layer_ids = {};
        ml.get_arr(key, layer_ids, true);
        for (uint32_t i = 0; i < hparams.dflash_n_target_layers; ++i) {
            if (layer_ids[i] < 0) {
                throw std::runtime_error(format("dflash: %s contains negative layer id %d", key.c_str(), layer_ids[i]));
            }
            hparams.dflash_target_layer_ids[i] = (uint32_t) layer_ids[i];
        }
    } else {
        std::array<uint32_t, 8> layer_ids = {};
        ml.get_arr(key, layer_ids, true);
        for (uint32_t i = 0; i < hparams.dflash_n_target_layers; ++i) {
            hparams.dflash_target_layer_ids[i] = layer_ids[i];
        }
    }

    for (uint32_t i = 0; i < hparams.dflash_n_target_layers; ++i) {
        const uint32_t id = hparams.dflash_target_layer_ids[i];
        for (uint32_t j = 0; j < i; ++j) {
            if (hparams.dflash_target_layer_ids[j] == id) {
                throw std::runtime_error(format(
                    "dflash: %s contains duplicate layer id %u",
                    key.c_str(),
                    id));
            }
        }
    }

    return true;
}

static void validate_dflash_hparams(llama_hparams & hparams, llm_arch arch) {
    if (hparams.dflash_block_size <= 1) {
        throw std::runtime_error(format("%s: dflash block_size must be > 1", llama_model_arch_name(arch)));
    }
    if (hparams.dflash_n_target_layers == 0) {
        throw std::runtime_error(format("%s: dflash target_layer_ids are required", llama_model_arch_name(arch)));
    }

    // DFlash feature width is target-model specific. Keep the serialized metadata intact here
    // and validate it against the live target model during DFlash init.

    if (hparams.dflash_n_target_features == 0) {
        throw std::runtime_error(format(
            "%s: dflash n_target_features must be > 0",
            llama_model_arch_name(arch)));
    }
    if (hparams.dflash_n_target_features % hparams.dflash_n_target_layers != 0) {
        throw std::runtime_error(format(
            "%s: dflash n_target_features=%u must be divisible by n_target_layers=%u",
            llama_model_arch_name(arch),
            hparams.dflash_n_target_features,
            hparams.dflash_n_target_layers));
    }
}

// SOURCE-READ: llama.cpp #29042 (542348a35c9e, MIT) llama_model_base::load_swa_pattern -- array first,
// else a scalar period. Here a free function so the legacy switch below and the models/ classes share it.
// get_arr does not check the length against n_layer (an array written over n_layer incl. nextn layers
// must load into an arch that sizes by n_layer_eff()), and the array is read into a zeroed copy so a
// short array cannot leave stale flags behind.
bool llm_read_swa_pattern(llama_model_loader & ml, llama_hparams & hparams, bool dense_first) {
    decltype(hparams.is_swa_impl) flags{};
    if (ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, flags, false)) {
        hparams.is_swa_impl = flags;
        return true;
    }
    uint32_t n_pattern = 0;
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern, false)) {
        hparams.set_swa_pattern(n_pattern, dense_first);
        return true;
    }
    return false;
}

void llm_load_swa_pattern(llama_model_loader & ml, llama_hparams & hparams, uint32_t n_pattern, bool dense_first) {
    if (!llm_read_swa_pattern(ml, hparams, dense_first)) {
        hparams.set_swa_pattern(n_pattern, dense_first);
    }
}


void llm_load_hparams(
        llama_model_loader & ml,
        llama_model & model, bool ignore_vocab) {
    auto & hparams = model.hparams;
    const gguf_context * ctx = ml.meta;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        model.gguf_kv.emplace(name, value);
    }

    ml.get_key(LLM_KV_BLOCK_COUNT,       hparams.n_layer);

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, model.name, false);

    // get hparams kv
    ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, hparams.n_vocab, !ignore_vocab);

    // everything past this point is not vocab-related
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,    hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,  hparams.n_embd);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_OUT, hparams.n_embd_out, false);
    if (hparams.n_embd_out == 0) {
        hparams.n_embd_out = hparams.n_embd;
    }

    // for classifier models (upstream llama-model.cpp load_hparams): the label count IS n_cls_out, which sizes
    // cls_out and the per-sequence rerank score. Read through the gguf API because this loader's get_arr has no
    // string-array case.
    {
        const int kid = gguf_find_key(ctx, ml.llm_kv(LLM_KV_CLASSIFIER_OUTPUT_LABELS).c_str());
        if (kid >= 0 && gguf_get_kv_type(ctx, kid) == GGUF_TYPE_ARRAY && gguf_get_arr_type(ctx, kid) == GGUF_TYPE_STRING) {
            const int n = gguf_get_arr_n(ctx, kid);
            model.classifier_labels.clear();
            for (int i = 0; i < n; ++i) {
                model.classifier_labels.emplace_back(gguf_get_arr_str(ctx, kid, i));
            }
            if (!model.classifier_labels.empty()) {
                hparams.n_cls_out = model.classifier_labels.size();
            }
        }
    }

    // for decision models (upstream llama-model.cpp load_hparams): the option label per output, read by /v1/systemone
    ml.get_arr(LLM_KV_DECISION_LABELS, model.decision_labels, false);

    ml.get_key(LLM_KV_EXPERT_COUNT,      hparams.n_expert,      false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used, false);

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
    }

    // zero-out the per-layer hparams
    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);
    std::fill(hparams.is_swa_impl.begin(),    hparams.is_swa_impl.end(),    0);
    std::fill(hparams.is_recr_impl.begin(), hparams.is_recr_impl.end(), false);
    std::fill(hparams.ttt_layer_impl.begin(), hparams.ttt_layer_impl.end(), false);
    // In-Place TTT has a graph route on qwen3 only (build_qwen3.cpp calls build_ttt_ffn). Another arch carrying "<arch>.ttt.chunk" would load
    // the tensors and silently run as the plain model, so refuse it. Row S69-FORK-INPLACE-TTT-OTHER-ARCHS-2026-10-26-01.
    if (model.arch != LLM_ARCH_QWEN3) {
        uint32_t ttt_chunk_probe = 0;
        if (ml.get_key(LLM_KV_TTT_CHUNK, ttt_chunk_probe, false)) {
            throw std::runtime_error(format("ttt: arch %s has no TTT graph route (only qwen3 does); the GGUF's ttt keys would be ignored", llm_arch_name(model.arch)));
        }
    }
    // -1 = no deepstack input for that layer. The src/models granite classes count every entry >= 0 as a deepstack
    // layer and inject an input view there, so the value-initialised zeros read as "deepstack index 0" on every layer
    // (a text granite_swa GGUF got n_embd_inp = 2 * n_embd). SOURCE-READ: upstream 64086f2b2 src/llama-model.cpp:1103 (MIT).
    std::fill(hparams.deepstack_mapping_arr.begin(), hparams.deepstack_mapping_arr.end(), -1);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    // by default assume that the sliding-window layers use the same scaling type as the non-sliding-window layers
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_MSCALE, hparams.rope_yarn_mscale, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_TRUNCATE, hparams.rope_yarn_truncate, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);

        hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false);

        // sanity check for n_rot (optional)
        hparams.n_rot = hparams.n_embd_head_k_full;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);

        if (model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_FALCON || model.arch == LLM_ARCH_BITNET_25 || model.arch == LLM_ARCH_BITNET_B158 || model.arch == LLM_ARCH_DECI) {
            if (hparams.n_rot != hparams.n_embd_head_k_full) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd_head_k_full));
            }
        }
    } else {
        hparams.n_rot = 0;
        hparams.n_embd_head_k_full = 0;
        hparams.n_embd_head_v_full = 0;
    }

    {
        hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
        hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa, false);
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa, false);

        hparams.n_rot_swa = hparams.n_rot;
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    }

    // arch-specific KVs
    // A LiLiCorr draft is an LLM_ARCH_DFLASH GGUF whose model object is the src/models class (llama_model_create_dflash_class):
    // its own load_arch_hparams reads the keys, so it takes the class branch (default) below, not the legacy DFlash case.
    const bool dflash_class = model.arch == LLM_ARCH_DFLASH && dynamic_cast<llama_model_base *>(&model) != nullptr;
    switch (dflash_class ? LLM_ARCH_UNKNOWN : model.arch) {
        case LLM_ARCH_K2_HORIZON:
            {
                // Common attention
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // MoVA (Mixture-of-Values Attention) — optional for dense models
                ml.get_key(LLM_KV_ATTENTION_GROUPNORM_GROUPS,   hparams.n_norm_groups);
                ml.get_key(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      hparams.n_value_expert, false);
                ml.get_key(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, hparams.n_value_expert_used, false);
                if (hparams.n_value_expert > 0) {
                    hparams.f_norm_group_eps = hparams.f_norm_rms_eps;
                }

                // MoE expert FFN params
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);

                // Dense/MoE layer split
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead, false);

                // Model type detection
                if (hparams.n_expert > 0) {
                    // MoE variant
                    switch (hparams.n_layer) {
                        case 48: model.type = e_model::MODEL_36B_A4B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                } else {
                    // Dense variant
                    switch (hparams.n_layer) {
                        case 28: model.type = e_model::MODEL_1B; break;
                        case 36: model.type = hparams.n_embd == 2560 ? e_model::MODEL_4B : e_model::MODEL_7B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                }
            } break;
        case LLM_ARCH_LLAMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 8) {
                    switch (hparams.n_layer) {
                        case 32: model.type = e_model::MODEL_8x7B; break;
                        case 56: model.type = e_model::MODEL_8x22B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                } else {
                    switch (hparams.n_layer) {
                        case 22: model.type = e_model::MODEL_1B; break;
                        case 26: model.type = e_model::MODEL_3B; break;
                        // granite uses a vocab with len 49152
                        case 32: model.type = hparams.n_vocab == 49152 ? e_model::MODEL_3B : (hparams.n_vocab < 40000 ? e_model::MODEL_7B : e_model::MODEL_8B); break;
                        case 36: model.type = e_model::MODEL_8B; break; // granite
                        case 40: model.type = e_model::MODEL_13B; break;
                        case 48: model.type = e_model::MODEL_34B; break;
                        case 60: model.type = e_model::MODEL_30B; break;
                        case 80: model.type = hparams.n_head() == hparams.n_head_kv() ? e_model::MODEL_65B : e_model::MODEL_70B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                }
            } break;
        case LLM_ARCH_EAGLE3:
            {
                // S69 #5/#32b — EAGLE3 draft head (PR #18039 port). Common dims (n_embd=4096,
                // n_layer=1, n_head=32, n_head_kv=8, n_ff=12288) auto-load from standard KV above.
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key_or_arr(LLM_KV_TARGET_LAYERS, hparams.target_layer_ids, 3, false); // reuses array<int,4> instantiation
                ml.get_key(LLM_KV_TARGET_HIDDEN_SIZE,   hparams.n_embd_tgt,          false);
                ml.get_key(LLM_KV_NORM_BEFORE_RESIDUAL, hparams.norm_before_residual, false);
                ml.get_key(LLM_KV_DRAFT_VOCAB_SIZE,     hparams.n_draft_vocab,       false);
                // No converter writes eagle3.draft_vocab_size: not this tree's conversion/llama.py, and not upstream's.
                // MEASURED 2026-10-18 (engsrv-1018) on the AngelSlim Qwen3-8B EAGLE3 draft, converted by this tree:
                // n_draft_vocab=0, so output.weight was expected at {4096, 151936} and the load failed on {4096, 32000}.
                // Take the size the way upstream does. SOURCE-READ: ggml-org/llama.cpp PR #18039 (MIT), eagle3 tensor
                // creation: n_draft_vocab = d2t->ne[0] when the gguf has a d2t. Otherwise use output.weight's ne[1].
                if (hparams.n_draft_vocab == 0) {
                    if (const ggml_tensor * d2t_meta = ml.get_tensor_meta("d2t")) {
                        hparams.n_draft_vocab = (uint32_t) d2t_meta->ne[0];
                    } else if (const ggml_tensor * out_meta = ml.get_tensor_meta("output.weight")) {
                        hparams.n_draft_vocab = (uint32_t) out_meta->ne[1];
                    }
                }
                // S69 #33: EAGLE3 has ASYMMETRIC vocab — input reuses the target tokenizer (151936) but the
                // draft head OUTPUTS draft-vocab (n_draft_vocab=32000). The logits buffer/read use hparams.n_vocab,
                // so it MUST equal the output width (32000) or the logits read OOBs (ggml-backend.cpp:225). Input
                // embeddings use the external target tok_embd, not n_vocab, so this only resizes the output path.
                if (hparams.n_draft_vocab > 0) hparams.n_vocab = hparams.n_draft_vocab;
                model.type = e_model::MODEL_UNKNOWN;   // 0.6B-class draft head; no dedicated enum
                LLAMA_LOG_INFO("%s: EAGLE3 hparams loaded n_layer=%u n_head=%u n_head_kv=%u n_embd=%u "
                               "n_ff=%u n_embd_tgt=%u n_draft_vocab=%u norm_before_residual=%d\n", __func__,
                               hparams.n_layer, hparams.n_head(0), hparams.n_head_kv(0), hparams.n_embd,
                               hparams.n_ff(0), hparams.n_embd_tgt, hparams.n_draft_vocab,
                               (int) hparams.norm_before_residual);
            } break;
        case LLM_ARCH_LLAMA4:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,   hparams.n_moe_layer_step);
                hparams.n_swa_pattern = 4;    // pattern: 3 chunked - 1 full
                // ADDED 2026-09-09, slice 32 -- see the note in the COHERE2 case above. LLAMA4's
                // built path reads n_swa_pattern (src/graphs/build_llama.cpp:44) and does chunked
                // masking from n_attn_chunk (src/llama.cpp:5634), so nothing is wrong today; but
                // is_swa_impl was empty, and every consumer of hparams.is_swa() -- src/models/*,
                // llama-kv-cache-iswa.cpp, and llama_model::get_rope_freq_base/_freq_scale -- would
                // read "no layer is sliding". Upstream populates it at src/models/llama4.cpp.
                // saverrest-1019: the period is the DEFAULT now, as upstream does since #29042 (src/models/llama4.cpp,
                // load_swa_pattern(ml, 4); MIT): a sliding_window_pattern key, array or period, wins. build_llama.cpp
                // reads hparams.is_swa(il) for LLAMA4, so the graph follows the key too; n_swa_pattern above stays
                // only as the "build the SWA mask at all" switch there. A file without the key loads as before.
                llm_load_swa_pattern(ml, hparams, 4);
                hparams.n_attn_chunk  = 8192; // should this be a gguf kv? currently it's the same for Scout and Maverick
                hparams.n_swa = 1; // TODO @ngxson : this is added to trigger the SWA branch (we store the chunked attn mask in the SWA tensor), will need to clean this up later
                // The pattern above makes is_swa_any() true, and llama_model::create_memory() (called unconditionally since 2026-09-29) asserts
                // !is_swa_any() when swa_type is NONE -- so every llama4 load aborted at llama-model.cpp:4244. Pair the pattern with its type, as
                // src/models/llama4.cpp does. n_swa stays 1 on purpose: the legacy graph path reads it as the SWA trigger and as the FA window
                // argument (build_llama.cpp:53-55), and the real chunk size lives in n_attn_chunk.
                hparams.swa_type = LLAMA_SWA_TYPE_CHUNKED;

                switch (hparams.n_expert) {
                    case 16:  model.type = MODEL_17B_16E; break;
                    case 128: model.type = MODEL_17B_128E; break;
                    default:  model.type = MODEL_UNKNOWN;
                }

                if (model.type == MODEL_17B_128E) {
                    hparams.use_kq_norm = false;
                }
            } break;
        case LLM_ARCH_DECI:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
                    case 162: model.type = e_model::MODEL_405B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINICPM:
            {
                // Defaults = the MiniCPM-2B constants (scale_emb 12, scale_depth 1.4, dim_model_base 256) the graph used to hardcode,
                // so a GGUF without the keys behaves exactly as before. The converter (conversion/minicpm.py) writes the real ones:
                // embedding_scale = scale_emb, residual_scale = scale_depth/sqrt(n_layer), logit_scale = hidden/dim_model_base
                // (the graph DIVIDES by logit_scale). Row S69-ENGREV-IPEXLLM-MINICPM-SCALARS-IGNORED-BY-GRAPH-2026-10-27-01.
                hparams.f_embedding_scale = 12.0f;
                hparams.f_residual_scale  = 1.4f / sqrtf(float(hparams.n_layer));
                hparams.f_logit_scale     = hparams.n_embd ? (float(hparams.n_embd) / 256.0f) : 1.0f;

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
                ml.get_key(LLM_KV_RESIDUAL_SCALE,  hparams.f_residual_scale,  false);
                ml.get_key(LLM_KV_LOGIT_SCALE,     hparams.f_logit_scale,     false);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_2B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GROK:
            {
                // defaults for old GGUFs
                hparams.yarn_beta_fast = 8.0f;
                hparams.f_logit_scale = 0.5773502691896257f;
                hparams.f_embedding_scale = 78.38367176906169f;
                hparams.f_attn_out_scale = 0.08838834764831845f;
                hparams.f_attn_logit_softcapping = 30.0f;
                hparams.f_router_logit_softcapping = 30.0f;
                // no final_logit_softcapping in grok-1
                hparams.f_final_logit_softcapping = 0.0f;

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale, false);
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
                ml.get_key(LLM_KV_ATTENTION_OUTPUT_SCALE, hparams.f_attn_out_scale, false);
                ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping, false);
                ml.get_key(LLM_KV_ROUTER_LOGIT_SOFTCAPPING, hparams.f_router_logit_softcapping, false);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

                ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_LENGTH, hparams.attn_temp_length, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR, hparams.yarn_ext_factor, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, hparams.yarn_attn_factor, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, hparams.yarn_beta_fast, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW, hparams.yarn_beta_slow, false);

                switch (hparams.n_layer) {
                    case 64: model.type = e_model::MODEL_314B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_FALCON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 60: model.type = e_model::MODEL_40B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                if (model.type == e_model::MODEL_13B) {
                    // TODO: become GGUF KV parameter
                    hparams.f_max_alibi_bias = 8.0f;
                }
            } break;
        case LLM_ARCH_STARCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 36: model.type = e_model::MODEL_3B; break;
                    case 42: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_15B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_REFACT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_1B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);

                switch (hparams.n_layer) {
                    case 3:
                        model.type = e_model::MODEL_17M; break; // bge-micro
                    case 6:
                        model.type = e_model::MODEL_22M; break; // MiniLM-L6
                    case 12:
                        switch (hparams.n_embd) {
                            case 384: model.type = e_model::MODEL_33M; break; // MiniLM-L12, bge-small
                            case 768: model.type = e_model::MODEL_109M; break; // bge-base
                        } break;
                    case 24:
                        model.type = e_model::MODEL_335M; break; // bge-large
                }
            } break;
        case LLM_ARCH_MODERNBERT:
            {
                // bidirectional pre-norm GeGLU encoder. No token-type embd. Mean pooling. Per-layer RoPE theta
                // (global every 3rd = 160000, else 10000) applied inside build_modernbert; eps from gguf.
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps, false);
                if (hparams.f_norm_eps == 0.0f) hparams.f_norm_eps = 1e-5f;
                hparams.causal_attn = false;
                ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);
                if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
                    hparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
                }
                switch (hparams.n_layer) {
                    case 22: model.type = e_model::MODEL_137M; break; // modernbert-base
                    case 28: model.type = e_model::MODEL_335M; break; // modernbert-large
                }
            } break;
        case LLM_ARCH_JINA_BERT_V2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                // optional, as upstream reads it: ggml-org/models jina-reranker-v1-tiny-en has no such key
                // and failed to load here with "key not found in model: jina-bert-v2.pooling_type"
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);
                hparams.f_max_alibi_bias = 8.0f;

                switch (hparams.n_layer) {
                    case 4: model.type = e_model::MODEL_33M; break; // jina-embeddings-small
                    case 12: model.type = e_model::MODEL_137M; break; // jina-embeddings-base
                }
            } break;
        case LLM_ARCH_NOMIC_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false); // optional, as upstream

                if (hparams.n_layer == 12 && hparams.n_embd == 768) {
                    model.type = e_model::MODEL_137M;
                }
            } break;
        case LLM_ARCH_BLOOM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 30:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            case 4096: model.type = e_model::MODEL_7B; break;
                        } break;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_MPT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,      hparams.f_clamp_kqv, false);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_30B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STABLELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_12B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_QWEN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2VL:
            {
                ml.get_mrope_sections(hparams.rope_sections, nullptr);  // always-mrope arch: never all-zero
            }
            // fall through
        case LLM_ARCH_QWEN2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_5B : e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = hparams.n_head() == 20 ? e_model::MODEL_4B : e_model::MODEL_13B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_A2_7B; break;
                    case 28: model.type = e_model::MODEL_57B_A14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;

        case LLM_ARCH_QWEN3:
            {
                ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                load_ttt_params(ml, hparams);
                switch (hparams.n_layer) {
                    case 28: model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_6B : e_model::MODEL_1_7B; break;
                    case 36: model.type = hparams.n_embd == 2560 ? e_model::MODEL_4B : e_model::MODEL_8B; break;
                    case 40: model.type = e_model::MODEL_14B; break;
                    case 64: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3VL:
            {
                load_qwen3vl_deepstack_layers(ml, hparams);
                ml.get_mrope_sections(hparams.rope_sections, &llama_model_loader::MROPE_DEFAULT_QWEN3VL);  // always-mrope arch: never all-zero
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: model.type = e_model::MODEL_1_7B; break;
                    case 36: model.type = hparams.n_embd == 2560 ? e_model::MODEL_4B : e_model::MODEL_8B; break;
                    case 64: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
                // since vision model stacks deepstack features along feature dim
                // we also create a fake "n_embd" for text model to be the main embd + deepstack embds
                hparams.n_embd *= hparams.n_deepstack_layers + 1;
                hparams.n_embd_inp_impl = hparams.n_embd;   // the widened n_embd IS the input width; see qwen3vl_pin_input_width
            } break;
        case LLM_ARCH_QWEN3MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_30B_A3B; break;
                    case 94: model.type = e_model::MODEL_235B_A22B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MELLUM:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);

                if (hparams.n_swa > 0) {
                    // SOURCE-READ upstream/llama.cpp/src/models/mellum.cpp:9 (MIT): STANDARD only when a window is configured
                    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
                    hparams.rope_freq_scale_train_swa = 1; //hparams.rope_freq_scale_train;

                    // array or period (#29042); absent -> period 4, i.e. is_swa = (i + 1) % 4 != 0 as before
                    llm_load_swa_pattern(ml, hparams, 4);

                    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
                }

                switch (hparams.n_layer) {
                    case 28: model.type = e_model::MODEL_12B_A2_5B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3NEXT:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // Upstream convention: every 4th layer is full attention, others are recurrent.
                for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                    hparams.is_recr_impl[i] = ((i + 1) % 4 != 0);
                }

                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_80B_A3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN4EXP:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

                ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
                ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);

                // bad metadata must be catchable: a zero here otherwise reaches a modulo, a reshape or a
                // GGML_ASSERT that aborts the whole process. SOURCE-READ upstream src/models/qwen4exp.cpp:10-14,
                // 38-61 @ 95887577a (MIT); hc_count <= 1 is refused by transformers/vLLM/SGLang too (:47-52).
                auto require_nonzero = [&](llm_kv kid, uint32_t v) {
                    if (v == 0) {
                        throw std::runtime_error(format("qwen4exp: %s must be greater than zero", ml.llm_kv(kid).c_str()));
                    }
                };
                require_nonzero(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                require_nonzero(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                require_nonzero(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                require_nonzero(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                require_nonzero(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
                require_nonzero(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
                require_nonzero(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
                require_nonzero(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
                require_nonzero(LLM_KV_HYPER_CONNECTION_LOW_RANK,    hparams.hc_low_rank);
                if (hparams.dsv4_hc_mult <= 1) {
                    throw std::runtime_error(format("qwen4exp: %s must be greater than one, got %u",
                            ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
                }

                // the MTP handover is the wide pre-mixer residual, so the width must not depend on an appended NextN block
                if (hparams.dsv4_hc_mult > 1) {
                    const uint32_t wide = hparams.n_embd * hparams.dsv4_hc_mult;
                    if (hparams.n_embd_out != hparams.n_embd && hparams.n_embd_out != wide) {
                        throw std::runtime_error("qwen4exp: embedding_length_out must equal n_embd * hyper-connection count");
                    }
                    hparams.n_embd_out = wide;
                }

                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                if (hparams.nextn_predict_layers >= hparams.n_layer) {
                    throw std::runtime_error("qwen4exp: nextn_predict_layers must be smaller than block_count");
                }

                {
                    uint32_t full_attn_interval = 4;
                    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
                    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                        hparams.is_recr_impl[i] = ((i + 1) % full_attn_interval != 0);
                    }
                    // the MTP tail is a full-attention (QSA) layer regardless of the interval pattern
                    for (uint32_t i = hparams.n_layer - hparams.nextn_predict_layers; i < hparams.n_layer; ++i) {
                        hparams.is_recr_impl[i] = false;
                    }
                }

                {
                    // block_count includes the nextn tail, but the converter's ratio array covers
                    // only the main layers; the tail inherits the last main QSA layer's ratio
                    const uint32_t n_main = hparams.n_layer - hparams.nextn_predict_layers;
                    uint32_t n_ratios = 0;
                    if (ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_ratios, false) && n_ratios >= n_main) {
                        std::vector<uint32_t> ratios;
                        ml.get_arr(ml.llm_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS), ratios);
                        const uint32_t n_copy = std::min<uint32_t>(n_ratios, hparams.n_layer);
                        std::copy_n(ratios.begin(), n_copy, hparams.dsv4_compress_ratios.begin());
                        for (uint32_t i = n_copy; i < hparams.n_layer; ++i) {
                            hparams.dsv4_compress_ratios[i] = ratios[n_main - 1];
                        }
                    }
                }

                // ple.layers is absent when the model carries no n-gram table, which leaves
                // the module inert
                {
                    uint32_t n_ple_layers = 0;
                    if (ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple_layers, false) && n_ple_layers > 0) {
                        // hparams hold ONE set of hash constants, so a second PLE module would silently reuse the
                        // first one's multipliers/offsets (upstream qwen4exp.cpp:73-77 refuses it the same way)
                        if (n_ple_layers != 1) {
                            throw std::runtime_error(format("qwen4exp: %s lists %u layers, but only one PLE layer is supported",
                                    ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple_layers));
                        }
                        std::vector<uint32_t> ple_layers;
                        ml.get_arr(ml.llm_kv(LLM_KV_PLE_LAYERS), ple_layers);
                        for (uint32_t il : ple_layers) {
                            if (il >= hparams.n_layer) {
                                throw std::runtime_error(format("qwen4exp: PLE layer index %u is outside block count %u", il, hparams.n_layer));
                            }
                            hparams.ple_layer_arr[il] = true;
                        }
                        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
                        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
                        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
                        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.ple_head_dim);
                        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id, false);
                        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);

                        require_nonzero(LLM_KV_PLE_CONV_KERNEL,            hparams.ple_conv_kernel);
                        require_nonzero(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.ple_head_dim);
                        // an n-gram of 1 has no predecessors: zero heads, an empty gather (upstream :97-102)
                        if (hparams.ple_ngram_size < 2) {
                            throw std::runtime_error(format("qwen4exp: PLE n-gram size %u is below 2", hparams.ple_ngram_size));
                        }
                        hparams.ple_n_heads = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
                        if (hparams.ple_n_heads == 0) {
                            throw std::runtime_error("qwen4exp: PLE heads per n-gram must be greater than zero");
                        }
                        if (hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
                            throw std::runtime_error("qwen4exp: PLE geometry exceeds the supported bounds");
                        }

                        std::vector<uint64_t> mults, offs, vocabs;
                        ml.get_arr(ml.llm_kv(LLM_KV_PLE_LAYER_MULTIPLIERS), mults);
                        ml.get_arr(ml.llm_kv(LLM_KV_PLE_HEAD_OFFSETS),      offs);
                        ml.get_arr(ml.llm_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES),  vocabs);
                        // the derived geometry indexes these arrays, and a short one would leave
                        // zeros that reach a modulo in the host fill
                        if (mults.size()  < hparams.ple_ngram_size ||
                            offs.size()   < hparams.ple_n_heads ||
                            vocabs.size() < hparams.ple_n_heads) {
                            throw std::runtime_error("qwen4exp: the PLE arrays are shorter than the declared geometry");
                        }
                        for (size_t i = 0; i < mults.size()  && i < LLAMA_MAX_PLE_NGRAM; ++i) hparams.ple_layer_multipliers[i] = mults[i];
                        for (size_t i = 0; i < offs.size()   && i < LLAMA_MAX_PLE_HEADS; ++i) hparams.ple_head_offsets[i]      = offs[i];
                        for (size_t i = 0; i < vocabs.size() && i < LLAMA_MAX_PLE_HEADS; ++i) hparams.ple_head_vocab_sizes[i]  = vocabs[i];
                        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
                            if (hparams.ple_head_vocab_sizes[h] == 0) {
                                throw std::runtime_error("qwen4exp: a PLE head declares a zero vocabulary size");
                            }
                        }
                    }
                }

                switch (hparams.n_layer - hparams.nextn_predict_layers) {
                    case 48: model.type = e_model::MODEL_125B_A6B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN35MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

                ml.get_mrope_sections(hparams.rope_sections, &llama_model_loader::MROPE_DEFAULT_QWEN35);  // always-mrope arch: never all-zero

                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                // keep the two nextn spellings in step -- see llama-hparams.h
                hparams.sync_nextn_from_nextn_predict_layers();
                if (model.mtp) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                } else {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }

                // Load linear attention (gated delta net) parameters
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // Mark recurrent layers (linear attention layers)
                {
                    uint32_t full_attn_interval = 4;
                    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
                    const uint32_t n_main_layers = hparams.n_layer - hparams.nextn_predict_layers;
                    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                        if (i < n_main_layers) {
                            hparams.is_recr_impl[i] = ((i + 1) % full_attn_interval != 0);
                        } else {
                            hparams.is_recr_impl[i] = false;
                        }
                    }
                }

                switch (hparams.n_layer) {
                    case 40:
                    case 41:
                        model.type = e_model::MODEL_35B_A3B; break;
                    case 48: model.type = e_model::MODEL_122B_A10B; break;
                    case 60: model.type = e_model::MODEL_397B_A17B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN35:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_mrope_sections(hparams.rope_sections, &llama_model_loader::MROPE_DEFAULT_QWEN35);  // always-mrope arch: never all-zero

                // NextN/MTP parameters
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                // keep the two nextn spellings in step -- see llama-hparams.h
                hparams.sync_nextn_from_nextn_predict_layers();
                if (model.mtp) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                } else {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }

                // Load linear attention (gated delta net) parameters
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // Mark recurrent layers (linear attention layers)
                // MTP layers always use standard attention, not delta-net
                {
                    uint32_t full_attn_interval = 4;
                    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
                    const uint32_t n_main_layers = hparams.n_layer - hparams.nextn_predict_layers;
                    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                        if (i < n_main_layers) {
                            hparams.is_recr_impl[i] = ((i + 1) % full_attn_interval != 0);
                        } else {
                            hparams.is_recr_impl[i] = false;
                        }
                    }
                }

                switch (hparams.n_layer) {
                    case 24: // without MTP layer
                    case 25: // with MTP layer (24 main + 1 MTP)
                        model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_8B : e_model::MODEL_2B; break;
                    case 32: // without MTP layer
                    case 33: // with MTP layer (32 main + 1 MTP)
                        model.type = hparams.n_embd == 2560 ? e_model::MODEL_4B   : e_model::MODEL_9B; break;
                    case 64: // without MTP layer
                    case 65: // with MTP layer (64 main + 1 MTP)
                        model.type = e_model::MODEL_27B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3VLMOE:
            {
                load_qwen3vl_deepstack_layers(ml, hparams);
                ml.get_mrope_sections(hparams.rope_sections, &llama_model_loader::MROPE_DEFAULT_QWEN3VL);  // always-mrope arch: never all-zero
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_30B_A3B; break;
                    case 94: model.type = e_model::MODEL_235B_A22B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
                // since vision model stacks deepstack features along feature dim
                // we also create a fake "n_embd" for text model to be the main embd + deepstack embds
                hparams.n_embd *= hparams.n_deepstack_layers + 1;
                hparams.n_embd_inp_impl = hparams.n_embd;   // the widened n_embd IS the input width; see qwen3vl_pin_input_width
            } break;
        case LLM_ARCH_PHI2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PHI3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                // for backward compatibility ; see: https://github.com/ggerganov/llama.cpp/pull/8931
                if ((hparams.n_layer == 32 || hparams.n_layer == 40) && hparams.n_ctx_train == 4096) {
                    // default value for Phi-3-mini-4k-instruct and Phi-3-medium-4k-instruct
                    hparams.n_swa = 2047;
                } else if (hparams.n_layer == 32 && hparams.n_head_kv(0) == 32 && hparams.n_ctx_train == 131072) {
                    // default value for Phi-3-mini-128k-instruct
                    hparams.n_swa = 262144;
                } else if (hparams.n_layer == 40 && hparams.n_ctx_train == 131072) {
                    // default value for Phi-3-medium-128k-instruct
                    hparams.n_swa = 131072;
                }
                bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                if (!found_swa && hparams.n_swa == 0) {
                    throw std::runtime_error("invalid value for sliding_window");
                }
            } break;
        case LLM_ARCH_PLAMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GPT2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 12: model.type = e_model::MODEL_SMALL; break;
                    case 24: model.type = e_model::MODEL_MEDIUM; break;
                    case 36: model.type = e_model::MODEL_LARGE; break;
                    case 48: model.type = e_model::MODEL_XL; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CODESHELL:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 42: model.type = e_model::MODEL_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ORION:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_20B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GEMMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 18: model.type = e_model::MODEL_2B; break;
                    case 28: model.type = e_model::MODEL_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GEMMA2:
            {
                hparams.n_swa = 4096; // default value of gemma 2
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping, false);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);
                hparams.attn_soft_cap = true;

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_2B; break;
                    case 42: model.type = e_model::MODEL_9B; break;
                    case 46: model.type = e_model::MODEL_27B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GEMMA3:
            {
                hparams.n_swa_pattern = 6;
                // ADDED 2026-09-09, slice 32 -- see the note in the COHERE2 case above. Same gap:
                // the pattern was recorded in n_swa_pattern and never in is_swa_impl, so
                // hparams.is_swa(il) was false on every GEMMA3 layer. Upstream populates it at
                // src/models/gemma3.cpp. Proven behaviour-neutral by the pattern selftest.
                // saverrest-1019: 6 is the DEFAULT now (upstream #29042 src/models/gemma3.cpp: load_swa_pattern(ml, 6); MIT);
                // a sliding_window_pattern key wins, and build_gemma3.cpp reads hparams.is_swa(il), so the graph follows it.
                llm_load_swa_pattern(ml, hparams, 6);
                // swa_type ADDED 2026-09-14, and the line above is WHY it is needed. That change
                // was recorded as behaviour-neutral on the strength of the pattern selftest, which
                // proves set_swa_pattern(N) reproduces the modular formula -- true, and not the
                // property that mattered. Populating is_swa_impl makes hparams.is_swa_any() TRUE,
                // and the KV-cache selector branches on a DIFFERENT field:
                //     llama-model.cpp:3842   if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) ...
                //     llama-model.cpp:3907   else GGML_ASSERT(!hparams.is_swa_any());
                // With the pattern set and swa_type left NONE, those two disagree and the assert
                // fires. MEASURED: gemma-3-1b-it-Q8_0 could not be LOADED at all --
                //     llama-model.cpp:3907: GGML_ASSERT(!hparams.is_swa_any()) failed
                // Setting swa_type is how upstream pairs them (src/models/cohere2.cpp sets
                // swa_type at :4 and the pattern at :7); one without the other is the defect.
                // NOTE: LLM_ARCH_LLAMA4 (:1644) and LLM_ARCH_COHERE2 (:1642) received the same
                // set_swa_pattern call in that slice and are in the same state. They are NOT
                // changed here because no model of either was available to verify against, and an
                // unverified fix to a load path is how this one arrived.
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

                hparams.rope_freq_base_train_swa  = 10000.0f;
                hparams.rope_freq_scale_train_swa = 1.0f;

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_2B; break;
                    case 34: model.type = e_model::MODEL_4B; break;
                    case 48: model.type = e_model::MODEL_12B; break;
                    case 62: model.type = e_model::MODEL_27B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                hparams.f_attention_scale = model.type == e_model::MODEL_27B
                    ? 1.0f / std::sqrt(float(hparams.n_embd / hparams.n_head(0)))
                    : 1.0f / std::sqrt(float(hparams.n_embd_head_k_full));
            } break;
        case LLM_ARCH_GEMMA_EMBEDDING:
            {
                // S69-OLLAMA-FORMAT-EMBEDDINGGEMMA-316-VS-314-TENSORS-2026-09-28-01: legacy-path
                // port. SOURCE-READ: src/models/gemma-embedding.cpp:3-30 (new-gen path; that path
                // is unreachable end to end -- memory/project-fork-models-wired-but-no-dispatcher.md
                // and findings/S69_EMBEDGEMMA_MTP_BAILING_2026-09-29.md -- so this is a from-scratch
                // port of its field assignments into the legacy switch, not a call into it). Field
                // list, order and values below are copied from that read, not reinvented.
                // swa_type paired with set_swa_pattern exactly as the GEMMA3 case above requires
                // (see its comment at :1089-1104): one without the other asserts at load.
                hparams.swa_type = LLAMA_SWA_TYPE_SYMMETRIC;
                // array or period (#29042); absent -> period 6 as before
                llm_load_swa_pattern(ml, hparams, 6);

                hparams.causal_attn = false; // embeddings do not use causal attention

                // The GGUF carries gemma-embedding.pooling_type (embeddinggemma: 1 = MEAN). Without this read the
                // model loaded with pooling NONE (MEASURED: load log 'pooling type = 0'), so append_pooling never pooled.
                ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);

                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // applied only if the model was converted with --sentence-transformers-dense-modules
                ml.get_key(LLM_KV_DENSE_2_FEAT_IN,  hparams.dense_2_feat_in,  false);
                ml.get_key(LLM_KV_DENSE_2_FEAT_OUT, hparams.dense_2_feat_out, false);
                ml.get_key(LLM_KV_DENSE_3_FEAT_IN,  hparams.dense_3_feat_in,  false);
                ml.get_key(LLM_KV_DENSE_3_FEAT_OUT, hparams.dense_3_feat_out, false);

                GGML_ASSERT((hparams.dense_2_feat_in  == 0 || hparams.dense_2_feat_in  == hparams.n_embd) && "dense_2_feat_in must be equal to n_embd");
                GGML_ASSERT((hparams.dense_3_feat_out == 0 || hparams.dense_3_feat_out == hparams.n_embd) && "dense_3_feat_out must be equal to n_embd");

                switch (hparams.n_layer_eff()) {
                    case 24: model.type = e_model::MODEL_0_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
                hparams.f_attention_scale = 1.0f / std::sqrt(float(hparams.n_embd_head_k()));
            } break;
        case LLM_ARCH_GEMMA4:
            {
                // Was commented out. The pattern read below fills is_swa_impl, so with swa_type left NONE
                // llama_model::create_memory took the non-SWA arm and aborted on
                // GGML_ASSERT(!hparams.is_swa_any()) -- MEASURED 2026-09-28: gemma-4-E2B-it-Q4_0 could not
                // load. Upstream pairs them (src/models/gemma4.cpp:4). No legacy-path code reads swa_type
                // (llama.cpp, llama-build-context.cpp, graphs/: 0 readers), so the computed graph is unchanged.
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer);

                uint32_t n_kv_shared_layers = 0;
                ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);

                hparams.n_layer_kv_from_start = hparams.n_layer - (int32_t)n_kv_shared_layers;
                hparams.f_attention_scale     = 1.0f; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)
                hparams.f_final_logit_softcapping = 30.0f;

                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);
                ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
                ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

                switch (hparams.n_layer) {
                    case 35: model.type = e_model::MODEL_2B; break;
                    case 42: model.type = e_model::MODEL_4B; break; // to confirm: E4B or E5B?
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DIFFUSION_GEMMA:
            {
                // The pattern read below fills is_swa_impl, so is_swa_any() is true; leaving swa_type NONE tripped the post-load pairing
                // warning (llama-hparams.cpp, "marks sliding-window layers but leaves hparams.swa_type NONE"). Pair them as gemma4 does.
                // Safe for the computed graph: this arch fills its own mask before the generic paths (llama.cpp, the DIFFUSION_GEMMA block
                // in llama_set_inputs) and create_memory returns no memory for it, so nothing reads swa_type.
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer);

                // bidirectional decoder: the forward fills its own region-aware mask, so the model
                // is declared non-causal and never touches the KV cache.
                hparams.causal_attn           = false;
                hparams.n_layer_kv_from_start = hparams.n_layer;
                hparams.f_attention_scale     = 1.0f; // Gemma4 uses self.scaling = 1.0
                hparams.f_final_logit_softcapping = 30.0f;

                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
                ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

                // canvas_length splits the forward at P = n_tokens - canvas_length; REQUIRED, and a
                // zero here would silently turn the whole batch into "prompt" and denoise nothing.
                ml.get_key(std::string("diffusion.canvas_length"), hparams.diffusion_canvas_length, true);
                if (hparams.diffusion_canvas_length == 0) {
                    throw std::runtime_error("DiffusionGemma requires a positive diffusion.canvas_length");
                }

                switch (hparams.n_layer) {
                    case 30: model.type = e_model::MODEL_26B_A4B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GEMMA4_MTP:
        case LLM_ARCH_GEMMA4_ASSISTANT:
            {
                // The pattern read below fills is_swa_impl; with swa_type left NONE create_memory takes the non-SWA arm and asserts !is_swa_any()
                // (test-llama-archs -a gemma4_mtp, 2026-10-04; the post-load pairing check warned about exactly this). The ASSISTANT arm of
                // create_memory sits under swa_type != NONE, so it was unreachable without this. Pair as gemma4 does.
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                if (model.arch == LLM_ARCH_GEMMA4_MTP) {
                    ml.get_key(LLM_KV_MTP_BACKBONE_EMBEDDING_LENGTH, hparams.mtp_backbone_n_embd);
                    ml.get_key(LLM_KV_MTP_CENTROID_COUNT,            hparams.mtp_num_centroids, false);
                    ml.get_key(LLM_KV_MTP_CENTROID_TOP_K,            hparams.mtp_centroid_top_k, false);
                } else {
                    ml.get_key("gemma4-assistant.embedding_length_out", hparams.mtp_backbone_n_embd);
                    ml.get_key("gemma4-assistant.n_centroids",     hparams.mtp_num_centroids, false);
                    ml.get_key("gemma4-assistant.centroid_top_k",  hparams.mtp_centroid_top_k, false);
                }
                ml.get_key(LLM_KV_MTP_USE_ORDERED_EMBEDDINGS,    hparams.mtp_use_ordered_embeddings, false);

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,       hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer);
                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,             hparams.rope_freq_base_train_swa, false);

                hparams.n_layer_kv_from_start = hparams.n_layer;
                hparams.f_attention_scale     = 1.0f;

                switch (hparams.mtp_backbone_n_embd) {
                    case 5376: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DFLASH_DRAFT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,              hparams.dflash_block_size, false);
                ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID,           hparams.dflash_mask_token_id, false);
                ml.get_key(LLM_KV_DFLASH_N_TARGET_FEATURES,       hparams.dflash_n_target_features, false);
                ml.get_key(LLM_KV_DFLASH_BACKBONE_ROTARY_BASE,    hparams.dflash_backbone_rotary_base, false);
                load_dflash_target_layer_ids(ml, LLM_KV(model.arch)(LLM_KV_DFLASH_TARGET_LAYER_IDS), hparams, false);
                ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, hparams.f_attn_v_scale, false);
                ml.get_key(LLM_KV_DFLASH_LAGUNA, hparams.dflash_laguna, false);
                ml.get_key(LLM_KV_DFLASH_SELECTOR_TOP_K,          hparams.dflash_selector_top_k, false);
                if (hparams.dflash_laguna) {
                    ml.get_key(LLM_KV_ATTENTION_CAUSAL, hparams.causal_attn);
                }
                // DFlash drafts may be trained with sliding-window attention (for long-context).
                // Read the window + per-layer pattern so the SWA mask path activates; absent keys
                // leave n_swa=0 / is_swa_impl all-zero (dense behavior, unchanged).
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer, false);
                // pair the pattern with swa_type, or create_memory aborts on !is_swa_any() (same defect as llama4 / diffusion-gemma)
                if (hparams.n_swa > 0 && hparams.is_swa_any()) {
                    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                }
                validate_dflash_hparams(hparams, model.arch);

                hparams.n_layer_kv_from_start = hparams.n_layer;
                model.type = e_model::MODEL_UNKNOWN;
            } break;

        case LLM_ARCH_STARCODER2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 30: model.type = e_model::MODEL_3B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_15B; break;
                    case 52: model.type = e_model::MODEL_20B; break; // granite
                    case 88: model.type = e_model::MODEL_34B; break; // granite
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MAMBA:
            {
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24:
                        switch (hparams.n_embd) {
                            case 768: model.type = e_model::MODEL_SMALL; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 48:
                        switch (hparams.n_embd) {
                            case 1024: model.type = e_model::MODEL_MEDIUM; break;
                            case 1536: model.type = e_model::MODEL_LARGE; break;
                            case 2048: model.type = e_model::MODEL_XL; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 64:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_XVERSE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    case 80: model.type = e_model::MODEL_65B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COMMAND_R:
            {
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_35B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DBRX:
        {
            ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
            ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,      hparams.f_clamp_kqv);

            switch (hparams.n_layer) {
                case 40: model.type = e_model::MODEL_16x12B; break;
                default: model.type = e_model::MODEL_UNKNOWN;
            }
        } break;
        case LLM_ARCH_OLMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv, false);

                switch (hparams.n_layer) {
                    case 22: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OPENELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                case 16: model.type = e_model::MODEL_270M; break;
                case 20: model.type = e_model::MODEL_450M; break;
                case 28: model.type = e_model::MODEL_1B; break;
                case 36: model.type = e_model::MODEL_3B; break;
                default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GPTNEOX:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_USE_PARALLEL_RESIDUAL, hparams.use_par_res);
                switch (hparams.n_layer) {
                    case 6:
                        switch (hparams.n_ff()) {
                            case 512: model.type = e_model::MODEL_14M; break;
                            case 2048: model.type = e_model::MODEL_70M; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: model.type = e_model::MODEL_160M; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 16:
                        switch (hparams.n_ff()) {
                            case 8192: model.type = e_model::MODEL_1B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096: model.type = e_model::MODEL_410M; break;
                            case 8192: model.type = e_model::MODEL_1_4B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 32:
                        switch (hparams.n_ff()) {
                            case 10240: model.type = e_model::MODEL_2_8B; break;
                            case 16384: model.type = e_model::MODEL_6_9B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 36:
                        switch (hparams.n_ff()) {
                            case 20480: model.type = e_model::MODEL_12B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 44:
                        switch (hparams.n_ff()) {
                            case 24576: model.type = e_model::MODEL_20B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ARCTIC:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 128) {
                    switch (hparams.n_layer) {
                        case 35: model.type = e_model::MODEL_10B_128x3_66B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                } else {
                    model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_DEEPSEEK2:
            {
                int expected_head_size_k = model.arch == LLM_ARCH_DEEPSEEK2 ? 576 : 320;
                int expected_head_size_v = model.arch == LLM_ARCH_DEEPSEEK2 ? 512 : 256;
                if (hparams.n_head_kv() == 1) {
                    int n_nead_kv = hparams.n_gqa();
                    if (n_nead_kv%4 != 0 || hparams.n_embd_head_k(0) != expected_head_size_k || hparams.n_embd_head_v(0) != expected_head_size_v ||
                        hparams.n_rot != 64) {
                        LLAMA_LOG_ERROR("==========================================================================\n");
                        LLAMA_LOG_ERROR("Detected incompatible DeepSeek model without a known way to fix it.\n");
                        LLAMA_LOG_ERROR("Consider making your own ik_llama.cpp compatible model or\n");
                        LLAMA_LOG_ERROR("ask the model provider to make one for you,\n\n");
                        LLAMA_LOG_ERROR("Sorry, unknown model => cannot fix it => bailing out\n");
                        LLAMA_LOG_ERROR("==========================================================================\n");
                        GGML_ABORT("Fatal error");
                    }
                    LLAMA_LOG_INFO("================= Adjusted mainline llama.cpp MLA tensors to ik_llama.cpp\n");
                    for (auto& item : hparams.n_head_kv_arr) item = n_nead_kv;
                    hparams.n_embd_head_k_full = 192;
                    hparams.n_embd_head_v_full = 128;
                    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_full);
                    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_full);
                }
                bool is_lite = (hparams.n_layer == 27 || hparams.n_layer == 26) || (hparams.n_layer == 48 && hparams.n_vocab == 128256);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
                if (!is_lite) {
                    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
                }
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm, false);
                hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_TYPE_NONE;
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    // Older DeepSeek models from the 2.0/2.5 series may not have the experts gating function recorded in the GGUF.
                    // Such models use SOFTMAX as the experts gating function.
                    // The new (new as of this commit) GLM-4.7-Flash may also be missing the experts gating function.
                    // GLM-4.7-Flash uses SIGMOID as the experts gating function.
                    // Hence, we make the LLM_KV_EXPERT_GATING_FUNC entry optional, and set here if missing.
                    // We distinguish between GLM-4.7-Flash and DeepSeek-2/2.5 models by the number of layers.
                    // GLM-4.7-Flash has 47 layers (or 48, if an MTP layer is included in the GGUF).
                    hparams.expert_gating_func = hparams.n_layer == 47 || hparams.n_layer == 48 ?
                        LLM_EXPERT_GATING_FUNC_SIGMOID : LLM_EXPERT_GATING_FUNC_SOFTMAX;
                    LLAMA_LOG_INFO("================= Missing experts gating function -> set to %s\n",
                            llm_expert_gating_func_name(llm_expert_gating_func_type(hparams.expert_gating_func)));
                }
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false);

                switch (hparams.n_layer) {
                    case 27: model.type = e_model::MODEL_16B; break;
                    case 36: model.type = e_model::MODEL_119B_A6B; break;
                    case 47: model.type = e_model::MODEL_30B_A3B; break; // GLM-4.7-Flash
                    case 60: model.type = e_model::MODEL_236B; break;
                    case 61: model.type = e_model::MODEL_671B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OPENPANGU:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

                // openPangu routes with a sigmoid gate + e_score_correction bias
                hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_TYPE_NONE;
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }

                // DSA lightning indexer
                ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head,    false);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size, false);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k,     false);

                // mHC / Hyper-Connections + learned param sink
                ml.get_key(LLM_KV_OPENPANGU_MHC_NUM_STREAM,    hparams.mhc_num_stream);
                ml.get_key(LLM_KV_OPENPANGU_MHC_RECUR_NORM,    hparams.mhc_recur_norm);
                ml.get_key(LLM_KV_OPENPANGU_PARAM_SINK_NUMBER, hparams.param_sink_number);

                // NextN / MTP layers are appended at the end and skipped for base generation
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                // keep the two nextn spellings in step -- see llama-hparams.h
                hparams.sync_nextn_from_nextn_predict_layers();
                if (hparams.nextn_predict_layers > 0 && hparams.nextn_predict_layers < hparams.n_layer) {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }

                // DSA/SWA schedule: openpangu.swa_layers lists the sliding-window layer ids and
                // openpangu.sliding_window_list the per-entry window; the remaining base layers
                // are DSA (indexer + top-k, no window). The NextN/MTP layers appear in the SWA
                // list with their own (larger) window, used by the MTP graphs. Absent keys leave
                // is_swa_impl cleared = dense fallback (pre-DSA GGUFs keep working).
                {
                    std::vector<uint32_t> swa_ids, swa_windows;
                    const bool have_ids = ml.get_arr("openpangu.swa_layers",          swa_ids,     false);
                    const bool have_win = ml.get_arr("openpangu.sliding_window_list", swa_windows, false);
                    if (have_ids && have_win && swa_ids.size() == swa_windows.size()) {
                        const uint32_t n_base = hparams.n_layer > hparams.nextn_predict_layers
                                              ? hparams.n_layer - hparams.nextn_predict_layers : hparams.n_layer;
                        for (size_t i = 0; i < swa_ids.size(); ++i) {
                            const uint32_t il = swa_ids[i];
                            if (il >= hparams.n_layer) {
                                throw std::runtime_error(format("openpangu.swa_layers contains out-of-range layer %u", il));
                            }
                            hparams.is_swa_impl[il] = swa_windows[i] > 0 ? 1 : 0;
                            if (il < n_base) {
                                if (hparams.n_swa != 0 && hparams.n_swa != swa_windows[i]) {
                                    throw std::runtime_error("openpangu: non-uniform base sliding windows are not supported");
                                }
                                hparams.n_swa = swa_windows[i];
                            } else {
                                if (hparams.n_swa_mtp != 0 && hparams.n_swa_mtp != swa_windows[i]) {
                                    throw std::runtime_error("openpangu: non-uniform MTP sliding windows are not supported");
                                }
                                hparams.n_swa_mtp = swa_windows[i];
                            }
                        }
                    } else if (have_ids || have_win) {
                        LLAMA_LOG_WARN("%s: openpangu SWA schedule keys are inconsistent - keeping dense fallback\n", __func__);
                    }
                    // the graph derives head dims and the MoME conv slot width from layer 0
                    if (hparams.n_swa > 0 &&
                        (hparams.n_embd_head_k_swa != hparams.n_embd_head_k_full ||
                         hparams.n_embd_head_v_swa != hparams.n_embd_head_v_full ||
                         hparams.n_rot_swa         != hparams.n_rot)) {
                        throw std::runtime_error("openpangu: per-layer SWA head dimensions are not supported");
                    }
                }


                model.type = e_model::MODEL_UNKNOWN; // 92B-A6B (46 + 3 MTP layers)
            } break;
        case LLM_ARCH_CHATGLM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: model.type = e_model::MODEL_6B; break;
                    case 40: model.type = e_model::MODEL_9B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GLM4:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_9B; break;
                    case 61: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GLM4_MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // MoE parameters
                ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
                ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

                // Expert gating function (GLM4_MOE uses sigmoid)
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == 0) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }

                // NextN/MTP parameters
                if (model.mtp) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                }
                else {
                    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                }
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);
                // keep the two nextn spellings in step -- see llama-hparams.h
                hparams.sync_nextn_from_nextn_predict_layers();

                switch (hparams.n_layer) {
                    case 47: model.type = e_model::MODEL_106B_A12B; break; // GLM-4.5-Air (46 layers + 1 NextN layer)
                    case 93: model.type = e_model::MODEL_355B_A32B; break; // GLM-4.5 (92 layers + 1 NextN layer)
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BITNET:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // Two HF classes convert to this arch: 1bitLLM's BitnetForCausalLM (hidden_act silu) and Microsoft's
                // BitNetForCausalLM, bitnet-b1.58-2B-4T (hidden_act relu2). The converter writes bitnet.hidden_activation;
                // files without it predate that and are 1bitLLM conversions, so SiLU stays the default.
                hparams.llm_ffn_op = LLM_FFN_SILU;
                std::string hidden_act;
                if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
                    if (hidden_act == "relu2") {
                        hparams.llm_ffn_op = LLM_FFN_RELU_SQR;
                    } else if (hidden_act != "silu") {
                        throw std::runtime_error("bitnet: unsupported hidden_activation '" + hidden_act + "' (expected silu or relu2)");
                    }
                }

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BITNET_B158:
        case LLM_ARCH_BITNET_25:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 30: model.type = e_model::MODEL_2B; break; // bitnet2b_2501
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_T5:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);

                uint32_t dec_start_token_id;
                if (ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, dec_start_token_id, false)) {
                    hparams.dec_start_token_id = dec_start_token_id;
                }

                switch (hparams.n_layer) {
                    case 6:  model.type = e_model::MODEL_60M;  break; // t5-small
                    case 8:  model.type = e_model::MODEL_80M;  break; // flan-t5-small
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: model.type = e_model::MODEL_220M; break; // t5-base
                            case 2048: model.type = e_model::MODEL_250M; break; // flan-t5-base
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096:  model.type = e_model::MODEL_770M; break; // t5-large
                            case 2816:  model.type = e_model::MODEL_780M; break; // flan-t5-large
                            case 16384: model.type = e_model::MODEL_3B;   break; // t5-3b
                            case 5120:  model.type = e_model::MODEL_3B;   break; // flan-t5-xl
                            case 65536: model.type = e_model::MODEL_11B;  break; // t5-11b
                            case 10240: model.type = e_model::MODEL_11B;  break; // flan-t5-xxl
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_T5ENCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);
                model.type = e_model::MODEL_UNKNOWN;
            } break;
        case LLM_ARCH_JAIS:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1_3B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    /* TODO: add variants */
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_RESIDUAL_SCALE, hparams.f_residual_scale);
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale);
                ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_3B; break;
                    // Add additional layer/vocab/etc checks here for other model sizes
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COHERE2:
            {
                hparams.n_swa_pattern = 4;
                // ADDED 2026-09-09, slice 32. n_swa_pattern above is this fork's representation and
                // is read by src/graphs/build_llama.cpp:44 -- it stays. But hparams.is_swa_impl was
                // NEVER populated for this architecture, so hparams.is_swa(il) returned false for
                // every layer while src/graphs/build_cohere2.cpp:31 treated three layers in four as
                // sliding, from its own local constant. Upstream populates it here
                // (src/models/cohere2.cpp:7). BEHAVIOUR-NEUTRAL: src/llama-swa-pattern.selftest.cpp
                // proves set_swa_pattern(N) reproduces that formula exactly, 79 checks, 0 failed,
                // negative controls firing.
                // saverrest-1019: 4 is the DEFAULT now (upstream #29042 src/models/cohere2.cpp: load_swa_pattern(ml, 4); MIT);
                // a sliding_window_pattern key wins, and build_cohere2.cpp reads hparams.is_swa(il), so the graph follows it.
                llm_load_swa_pattern(ml, hparams, 4);
                // paired with the pattern as upstream does (SOURCE-READ upstream/llama.cpp/src/models/cohere2.cpp:4, MIT);
                // without it create_memory asserts !is_swa_any() -- reproduced by test-llama-archs -a cohere2 on 2026-10-04
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_8B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COHERE2_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
                // array or period (#29042: upstream cohere2moe load_swa_pattern(4, true)); the array is no
                // longer length-checked against n_layer, so one written over n_layer incl. nextn still loads
                llm_load_swa_pattern(ml, hparams, 4, true);
                // SOURCE-READ upstream/llama.cpp/src/models/cohere2moe.cpp:27 (MIT): the pattern is always paired with STANDARD
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                ml.get_key(LLM_KV_LOGIT_SCALE,                      hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,        hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,       hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,              hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,               hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }
                switch (hparams.n_layer) {
                    case 49: model.type = e_model::MODEL_30B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAILINGMOE2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,                hparams.n_expert_groups);
                ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT,           hparams.n_group_used);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);
                // keep the two nextn spellings in step -- see llama-hparams.h
                hparams.sync_nextn_from_nextn_predict_layers();

                // TODO: when MTP is implemented, this should probably be updated if needed
                hparams.n_layer_kv_from_start = static_cast<int32_t>(hparams.n_layer - hparams.nextn_predict_layers);

                switch (hparams.n_layer) {
                    case 20: model.type = MODEL_16B_A1B; break;
                    case 21: model.type = MODEL_16B_A1B; break;
                    case 32: model.type = MODEL_100B_A6B; break;
                    case 33: model.type = MODEL_100B_A6B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;


        case LLM_ARCH_GRANITE_HYBRID:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // the four Granite scales; identical handling to LLM_ARCH_GRANITE above
                ml.get_key(LLM_KV_LOGIT_SCALE,     hparams.f_logit_scale);
                ml.get_key(LLM_KV_RESIDUAL_SCALE,  hparams.f_residual_scale);
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale);
                ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale);

                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // This architecture's recurrent layers are MAMBA2, not mamba1. The flag is not
                // cosmetic: n_embd_k_s() and n_embd_v_s() treat a non-zero ssm_n_group as
                // "this is qwen3next" and would size a delta-net state. granitehybrid writes
                // ssm.group_count = 1, so it lands in that branch unless this is set, and the
                // resulting mis-sized cache would surface nowhere near its cause.
                hparams.ssm_mamba2 = true;

                // head_count_kv is written per layer and is 0 on the mamba2 layers, so the array
                // itself says which layers are recurrent -- the convention bailingmoe3, kimi-k3
                // and lfm2 already use, applied here to a fourth mechanism.
                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.is_recr_impl[il] = hparams.n_head_kv_arr[il] == 0;
                }

                // expert_count is 0 in the -h- dense files; every FFN is dense
                hparams.n_layer_dense_lead = hparams.n_layer;

                // keyed on n_embd, because n_layer 32 is shared with granite-3B above
                switch (hparams.n_embd) {
                    case  768: model.type = e_model::MODEL_350M; break;
                    case 1536: model.type = e_model::MODEL_1B;   break;
                    default:   model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_LFM2:
        case LLM_ARCH_LFM2MOE:
            {
                // LFM2MOE (ik_llama #2530, hand-ported onto this tree's own LFM2 case): same hybrid shortconv/GQA
                // layers, sparse FFN on every block past n_layer_dense_lead.
                const bool is_moe = model.arch == LLM_ARCH_LFM2MOE;

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

                // head_count_kv is written per layer and is 0 on the short-convolution layers,
                // so the array itself says which layers are recurrent -- the same convention
                // bailingmoe3 and kimi-k3 already use, applied to a different mechanism.
                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.is_recr_impl[il] = hparams.n_head_kv_arr[il] == 0;
                }

                if (is_moe) {
                    hparams.n_layer_dense_lead = 0;
                    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,  hparams.n_layer_dense_lead, false);
                    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func, false);
                    hparams.expert_weights_norm = true;
                    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
                    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);

                    if (hparams.n_expert == 0 || hparams.n_expert_used == 0) {
                        throw std::runtime_error("LFM2-MoE requires a non-zero expert count");
                    }
                    if (hparams.n_layer_dense_lead > hparams.n_layer) {
                        throw std::runtime_error("LFM2-MoE leading_dense_block_count exceeds the block count");
                    }
                } else {
                    // every layer's FFN is dense in the dense family
                    hparams.n_layer_dense_lead = hparams.n_layer;
                }

                // Reuse the existing recurrent-state allocator instead of adding a code path.
                // The hybrid allocator sizes kv_self.s_l[il] by n_embd_v_s(), which for a
                // non-qwen3next model is ssm_d_state * ssm_d_inner. A short convolution carries
                // (l_cache - 1) * n_embd floats, so these three assignments make n_embd_v_s()
                // and n_embd_k_s() both land on exactly that.
                // ssm_d_state is NOT a state dimension in any SSM sense here -- it is the
                // carried-column count. Named oddly on purpose rather than adding a parallel
                // field; anyone touching l_cache should re-check that identity holds.
                hparams.ssm_d_conv  = hparams.n_shortconv_l_cache;
                hparams.ssm_d_inner = hparams.n_embd;
                hparams.ssm_d_state = hparams.n_shortconv_l_cache > 0
                                    ? hparams.n_shortconv_l_cache - 1 : 0;

                // sizes keyed on n_ff, as mainline's own table does
                if (is_moe) {
                    switch (hparams.n_layer) {
                        case 24: model.type = e_model::MODEL_8B_A1B;  break;
                        case 40: model.type = e_model::MODEL_24B_A2B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                } else switch (hparams.n_ff()) {
                    case  4608: model.type = e_model::MODEL_350M; break;
                    case  6912: model.type = e_model::MODEL_700M; break;
                    case  8192: model.type = e_model::MODEL_1_2B; break;
                    case 10752: model.type = e_model::MODEL_2_6B; break;
                    default:    model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_INKLING:
            {
                // S69 P4. SOURCE-READ: upstream/unsloth-llama-inkling-raw/src__models__inkling.cpp:12-52
                // (llama_model_inkling::load_arch_hparams; unslothai/llama.cpp inkling-25731-b10775, MIT), translated to this
                // fork's legacy field spellings (n_layer_all -> n_layer, n_ff_exp_arr -> the scalar n_ff_exp: the converter
                // writes one expert width).
                //
                // DELIBERATELY NOT SET (decided in S69 P1, which landed the graph in src/graphs/build_inkling.cpp):
                //   is_recr_impl[il]  upstream marks EVERY layer recurrent AND attention (hybrid-iSWA memory). In this fork
                //                     is_recr_impl alone switches llama_model::cache_size() to a state-only row, which would
                //                     size the KV wrongly for layers that also attend. So NO layer is recurrent here; the
                //                     arch is hybrid (llm_arch_is_hybrid) and the packed [k|v|attn|mlp] conv-state row is
                //                     sized by hparams.n_embd_inkling_conv(il) beside each layer's K/V cells.
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                // visible iff pos_q - pos_k < n_swa (the query itself included)
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer);

                ml.get_key(LLM_KV_INKLING_D_REL,             hparams.inkling_d_rel);
                ml.get_key(LLM_KV_INKLING_REL_EXTENT,        hparams.inkling_rel_extent);
                ml.get_key(LLM_KV_INKLING_REL_EXTENT_SWA,    hparams.inkling_rel_extent_swa);
                ml.get_key(LLM_KV_INKLING_SHORTCONV_KERNEL,  hparams.n_shortconv_l_cache);
                ml.get_key(LLM_KV_INKLING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);

                float logit_scale_denom = 0.0f;
                ml.get_key(LLM_KV_INKLING_LOGIT_SCALE_DENOM, logit_scale_denom);
                if (logit_scale_denom == 0.0f) {
                    throw std::runtime_error("inkling: logit_scale_denom must be non-zero");
                }
                hparams.f_logit_scale = 1.0f / logit_scale_denom;

                ml.get_key(LLM_KV_INKLING_LOG_SCALING_N_FLOOR, hparams.inkling_log_n_floor,      false);
                ml.get_key(LLM_KV_INKLING_LOG_SCALING_ALPHA,   hparams.inkling_log_alpha,        false);
                ml.get_key(LLM_KV_INKLING_UNPADDED_VOCAB_SIZE, hparams.inkling_unpadded_n_vocab, false);

                if (hparams.n_shortconv_l_cache <= 1) {
                    throw std::runtime_error("inkling: shortconv_kernel must be > 1");
                }
                if (hparams.inkling_d_rel == 0) {
                    throw std::runtime_error("inkling: d_rel must be > 0");
                }
                if (hparams.inkling_rel_extent == 0 || hparams.inkling_rel_extent_swa == 0) {
                    throw std::runtime_error("inkling: rel_extent and rel_extent_swa must be > 0");
                }
                if (hparams.n_layer_dense_lead > hparams.n_layer) {
                    throw std::runtime_error("inkling: dense_block_count exceeds the block count");
                }
                if (hparams.n_layer_dense_lead < hparams.n_layer &&
                        (hparams.n_expert == 0 || hparams.n_expert_used == 0 || hparams.n_expert_shared == 0)) {
                    throw std::runtime_error("inkling: MoE blocks need expert_count, expert_used_count and expert_shared_count");
                }
                if (hparams.inkling_unpadded_n_vocab > hparams.n_vocab) {
                    throw std::runtime_error("inkling: unpadded_vocab_size exceeds vocab_size");
                }

                model.type = e_model::MODEL_UNKNOWN;
            } break;
        case LLM_ARCH_KIMI_K3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm);
                // this file declares no expert_group_count; one group is the ungrouped case
                hparams.n_expert_groups = 1;
                ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,                hparams.n_expert_groups, false);
                ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT,           hparams.n_group_used, false);
                // no separate shared-expert FFN length is written; the shared expert is sized
                // like a routed one, so fall back rather than throw on a valid file
                hparams.n_ff_shexp = hparams.n_ff_exp;
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);

                ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,             hparams.n_lora_q);
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,            hparams.n_lora_kv);
                ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,          hparams.n_embd_head_k_full);
                ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,        hparams.n_embd_head_v_full);

                ml.get_key(LLM_KV_SSM_CONV_KERNEL,                   hparams.ssm_d_conv);
                ml.get_key(LLM_KV_KDA_HEAD_DIM,                      hparams.ssm_d_state);
                // absent key means true: only the safe-gate formula is implemented here
                hparams.kda_safe_gate = true;
                ml.get_key(LLM_KV_KDA_SAFE_GATE,                     hparams.kda_safe_gate, false);
                if (!hparams.kda_safe_gate) {
                    throw std::runtime_error("kimi-k3: kda.safe_gate = false is not supported");
                }
                ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,              hparams.kda_gate_lower_bound, false);

                // the four keys this architecture adds over kimi-linear
                ml.get_key(LLM_KV_EXPERT_LATENT_LENGTH,              hparams.n_expert_latent);
                ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,               hparams.attn_res_block_size);
                ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,              hparams.situ_beta);
                ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA,       hparams.situ_linear_beta);

                // unslothai/llama.cpp PR 70 (src/models/kimi-k3.cpp): zero in either makes the graph divide by it or run the dense
                // path on latent weights. A throw is returned by the loader as a load error (upstream GGML_ASSERTs = abort).
                if (hparams.attn_res_block_size == 0) {
                    throw std::runtime_error("Kimi-K3 requires attn_res.block_size > 0");
                }
                if (hparams.n_expert_latent == 0) {
                    throw std::runtime_error("Kimi-K3 requires expert_latent_length > 0");
                }

                hparams.ssm_n_group = hparams.n_head();
                hparams.ssm_dt_rank = hparams.n_head();
                hparams.ssm_d_inner = hparams.ssm_d_state * hparams.ssm_dt_rank;

                // head_count_kv is written per layer and is 0 on the KDA (linear) layers, so the
                // array itself says which layers are recurrent -- same convention as bailingmoe3
                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.is_recr_impl[il] = hparams.n_head_kv_arr[il] == 0;
                }

                switch (hparams.n_layer) {
                    case 8:  model.type = e_model::MODEL_1B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAILINGMOE3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,                hparams.n_expert_groups);
                ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT,           hparams.n_group_used);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
                // Ling-3.0-tiny ships no NextN block, and its converters omit the key
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);
                // keep the two nextn spellings in step -- see llama-hparams.h
                hparams.sync_nextn_from_nextn_predict_layers();
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,            hparams.n_lora_kv);
                // Ling-3.0-flash sets q_lora_rank null and projects Q directly; Ling-3.0-tiny factorizes it
                ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,             hparams.n_lora_q, false);
                ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,           hparams.n_embd_head_k_full);
                ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,         hparams.n_embd_head_v_full);
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,                    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_KDA_HEAD_DIM,                       hparams.ssm_d_state);
                // absent key means true: only the safe-gate formula is implemented
                hparams.kda_safe_gate = true;
                ml.get_key(LLM_KV_KDA_SAFE_GATE,                      hparams.kda_safe_gate, false);
                if (!hparams.kda_safe_gate) {
                    throw std::runtime_error("bailingmoe3: kda.safe_gate = false is not supported");
                }
                ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,               hparams.kda_gate_lower_bound);
                // Ling 3.0 VL files carry mrope sections ([t, h, w] + padding); text-only files omit the key and keep
                // rope_sections all zero, so use_mrope() is false and llama_rope_type stays NORM (ik_llama 719f4aa5d).
                // Optional + tolerant of a 3-entry array, as for HUNYUAN_VL (DEC-S69-MROPE-SECTIONS-TOLERANT).
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,     hparams.rope_sections, 4, false, true);
                if (hparams.use_mrope()) {
                    // ggml_rope_multi asserts sum(sections) <= the rotary width at COMPUTE time (ggml-cpu ops.cpp
                    // ggml_mrope_cache_init), which aborts the process on the first decode. Refuse such a file at load
                    // with the numbers instead. Real VL files carry sum = n_rot/2 (one section entry per rotary pair).
                    const uint32_t sect_sum = hparams.rope_sections[0] + hparams.rope_sections[1] +
                                              hparams.rope_sections[2] + hparams.rope_sections[3];
                    if (sect_sum > hparams.n_rot) {
                        throw std::runtime_error(format("bailingmoe3: rope.dimension_sections sum to %u, more than the rotary width %u",
                                                        sect_sum, hparams.n_rot));
                    }
                }
                // Ling-3.0-tiny sets both limit lists null. 0 is the unclamped value where it is read.
                hparams.swiglu_clamp_exp.fill(0.0f);
                hparams.swiglu_clamp_shexp.fill(0.0f);
                ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,            hparams.swiglu_clamp_exp,        hparams.n_layer, false);
                ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,          hparams.swiglu_clamp_shexp, hparams.n_layer, false);

                // one converter writes the tensors but not the key, so the tensors decide.
                // the norm is 1-D and so cannot be transposed, unlike attn_q_a itself
                if (hparams.n_lora_q == 0) {
                    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                        const std::string probe = LLM_TN(LLM_ARCH_BAILINGMOE3)(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il);
                        if (const auto * meta = ml.get_tensor_meta(probe.c_str())) {
                            hparams.n_lora_q = meta->ne[0];
                            break;
                        }
                    }
                }

                hparams.ssm_n_group = hparams.n_head();
                hparams.ssm_dt_rank = hparams.n_head();
                hparams.ssm_d_inner = hparams.ssm_d_state * hparams.ssm_dt_rank;
                // believe the tensors: a stale count hides the last real layer
                if (hparams.nextn_predict_layers > 0) {
                    const std::string probe = LLM_TN(LLM_ARCH_BAILINGMOE3)(LLM_TENSOR_NEXTN_EH_PROJ, "weight",
                            hparams.n_layer - hparams.nextn_predict_layers);
                    if (ml.get_tensor_meta(probe.c_str()) == nullptr) {
                        hparams.nextn_predict_layers = 0;
                    }
                }
                hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;

                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.is_recr_impl[il] = hparams.n_head_kv_arr[il] == 0;
                }

                model.type = e_model::MODEL_UNKNOWN;
            } break;
        case LLM_ARCH_DOTS1:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                switch (hparams.n_layer) {
                    case 62: model.type = e_model::MODEL_142B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        {
            ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
            if (model.arch == LLM_ARCH_ERNIE4_5_MOE) {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP, hparams.n_moe_layer_step);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
            }

            switch (hparams.n_layer) {
            case 18: model.type = e_model::MODEL_0_3B; break;
            case 28: model.type = e_model::MODEL_21B_A3B; break;
            case 54: model.type = e_model::MODEL_300B_A47B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            }
        } break;
        case LLM_ARCH_HUNYUAN_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_80B_A13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OPENAI_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

                //TODO OAI_MOE: SWA
                //hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                //hparams.set_swa_pattern(2);

                // TODO: switch (hparams.n_layer)

            } break;
        case LLM_ARCH_MINIMAX_M2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false);

                switch (hparams.n_layer) {
                    case 62: model.type = e_model::MODEL_230B_A10B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINIMAX_M3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head,    false);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size, false);

                model.type = e_model::MODEL_UNKNOWN;
            } break;
        case LLM_ARCH_SMOLLM3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                hparams.n_no_rope_layer_step = 4;

                switch (hparams.n_layer) {
                    case 36: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_HYPERCLOVAX:
            {
                // S69-EXLLAMAV3-PICK-HYPERCLOVAX-2026-10-20-01. SOURCE-READ: transformers HyperCLOVAXConfig / HyperCLOVAXDecoderLayer
                // (modular_hyperclovax.py, Apache-2.0) and exllamav3 architecture/hyperclovax.py (MIT); behaviour, no code copied.
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                // The four muP scales use the Granite GGUF keys and the Granite meaning, with ONE difference the converter absorbs:
                // HyperCLOVAX *multiplies* the logits by `logits_scaling` where Granite divides by `logit_scale`, so the converter
                // stores 1/logits_scaling in LLM_KV_LOGIT_SCALE and this arch's graph keeps the Granite division (cur / f_logit_scale).
                // All four are optional: an absent key (0.0) means "not scaled", the HF defaults of 1.0 being the identity.
                ml.get_key(LLM_KV_LOGIT_SCALE,     hparams.f_logit_scale,     false);
                ml.get_key(LLM_KV_RESIDUAL_SCALE,  hparams.f_residual_scale,  false);
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
                ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale, false);

                model.type = e_model::MODEL_UNKNOWN;
            } break;
        case LLM_ARCH_MISTRAL3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_SCALE, hparams.f_attn_temp_scale, false);

                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST,   hparams.yarn_beta_fast, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   hparams.yarn_beta_slow, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL,     hparams.rope_yarn_log_mul, false);

                if (hparams.f_attn_temp_scale != 0.0f) {
                    hparams.n_attn_temp_floor_scale = hparams.n_ctx_orig_yarn;
                    if (hparams.n_attn_temp_floor_scale == 0) {
                        throw std::runtime_error("invalid n_ctx_orig_yarn for attention temperature scaling");
                    }
                }

                // TODO: this seems to be correct with the case of mscale == mscale_all_dims == 1.0f
                //       but may need further verification with other values
                if (hparams.rope_yarn_log_mul != 0.0f) {
                    float factor = 1.0f / hparams.rope_freq_scale_train;
                    float mscale = hparams.rope_yarn_mscale != 0.0f ? hparams.rope_yarn_mscale : 1.0f;
                    float mscale_all_dims = hparams.rope_yarn_log_mul;
                    hparams.yarn_attn_factor = llama_yarn_attn_factor_ratio(factor, mscale, mscale_all_dims);
                }

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_3B; break;
                    case 34: model.type = e_model::MODEL_8B; break;
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MIMO2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,   hparams.n_swa);
                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,         hparams.rope_freq_base_train_swa);
                ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE,      hparams.f_attn_v_scale, false);
                // the pattern read below makes is_swa_any() true, and create_memory asserts when swa_type is still NONE (test-llama-archs -a mimo2, 2026-10-04);
                // upstream pairs them (SOURCE-READ upstream/llama.cpp/src/models/mimo2.cpp:6, MIT)
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD; // which is the same as OpenAI
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer);
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);
                // ik_llama.cpp: when MTP is implemented this may need updating
                hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;

                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_310B_A15B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

            } break;
        case LLM_ARCH_SEED_OSS:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 64: model.type = e_model::MODEL_36B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STEP35:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                // keep the two nextn spellings in step -- see llama-hparams.h
                hparams.sync_nextn_from_nextn_predict_layers();
                if (hparams.nextn_predict_layers > hparams.n_layer) {
                    throw std::runtime_error(format("step35.nextn_predict_layers (%u) exceeds block_count (%u)",
                            hparams.nextn_predict_layers, hparams.n_layer));
                }
                hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;
                // paired with the SWA pattern read below; see MIMO2 above (SOURCE-READ upstream/llama.cpp/src/models/step35.cpp:6, MIT)
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                // MoE + SWA parameters
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
                // Step35 uses sigmoid gating by default (if not set in GGUF)
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                bool have_rfb_train_swa = ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer);
                if (!ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_COUNT_PER_LAYER, hparams.rope_dim_per_layer, hparams.n_layer, false)) {
                    for (int i = 0; i < hparams.n_layer; ++i) {
                        hparams.rope_dim_per_layer[i] = hparams.is_swa_impl[i] ? hparams.n_rot : hparams.n_rot/2;
                    }
                }
                // The following two parameters: one of the two versions must be present in the GGUF
                if (!ml.get_key_or_arr(LLM_KV_SWIGLU_LIMITS,        hparams.swiglu_clamp_exp,        hparams.n_layer, false)) {
                     ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp, hparams.n_layer, true);
                }
                if (!ml.get_key_or_arr(LLM_KV_SWIGLU_LIMITS_SHARED, hparams.swiglu_clamp_shexp, hparams.n_layer, false)) {
                     ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_clamp_shexp, hparams.n_layer, true);
                }
                // Optional: Step35-only gating for applying rope scaling (HF: yarn_only_types).
                // Default is 3 (apply on all layers) if the key is absent.
                //ml.get_key(format("%s.rope.scaling.apply_mask", ml.get_arch_name().c_str()),
                //    hparams.rope_scaling_apply_mask, false);
                //hparams.has_rope_freq_base_per_layer = ml.get_key_or_arr(
                //    format("%s.rope.freq_base_per_layer", ml.get_arch_name().c_str()),
                //    hparams.rope_freq_base_per_layer, hparams.n_layer, false);
                ml.get_key(format("%s.rope.scaling.apply_mask", ml.get_arch_name().c_str()),
                    hparams.rope_scaling_apply_mask, false);
                hparams.has_rope_freq_base_per_layer = ml.get_key_or_arr(LLM_KV_ROPE_FREQ_BASE_PER_LAYER,
                    hparams.rope_freq_base_per_layer, hparams.n_layer, false);
                GGML_ASSERT(hparams.has_rope_freq_base_per_layer || have_rfb_train_swa);
            } break;
        case LLM_ARCH_LAGUNA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_TYPE_NONE;
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,          hparams.n_layer_dense_lead, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,                hparams.n_expert_shared, false);
                // Older Laguna GGUFs encode one shared expert through the shared FFN length.
                if (hparams.n_expert_shared == 0 && hparams.n_ff_shexp > 0) {
                    hparams.n_expert_shared = 1;
                }
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
                hparams.rope_freq_scale_train_swa = 1.0f;
                // array or period (#29042: upstream laguna dense_first=true); absent -> head-count fallback below
                if (!llm_read_swa_pattern(ml, hparams, true)) {
                    // Laguna XS.2 alternates full-attention and SWA layers via per-layer head counts.
                    const uint32_t n_head_full = hparams.n_head(0);
                    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                        hparams.is_swa_impl[i] = hparams.n_head(i) != n_head_full;
                    }
                }

                const bool found_rope_dim     = ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT,     hparams.n_rot,     false);
                const bool found_rope_dim_swa = ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);

                // Laguna GGUFs store the number of scalar Q/K dimensions that ggml_rope_ext
                // rotates. Correct files carry those values explicitly. Some early public
                // XS.2 GGUFs omitted both keys, so fall back to the HF XS.2 layout only for
                // missing metadata: full-attention layers rotate half the head, SWA layers
                // rotate the full head. Explicit but wrong halved metadata still needs repair.
                if (hparams.n_swa > 0) {
                    // SOURCE-READ upstream/llama.cpp/src/models/laguna.cpp:36-37 (MIT): STANDARD only when a window is configured
                    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                    if (!found_rope_dim) {
                        hparams.n_rot = hparams.n_embd_head_k_full / 2;
                    }
                    if (!found_rope_dim_swa) {
                        hparams.n_rot_swa = hparams.n_embd_head_k_swa;
                    }
                }

                ml.get_key(LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR, hparams.yarn_ext_factor, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, hparams.yarn_attn_factor, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST,   hparams.yarn_beta_fast,   false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   hparams.yarn_beta_slow,   false);

                if (!ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_COUNT_PER_LAYER, hparams.rope_dim_per_layer, hparams.n_layer, false)) {
                    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                        hparams.rope_dim_per_layer[i] = hparams.is_swa_impl[i] ? hparams.n_rot_swa : hparams.n_rot;
                    }
                }

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_33B_A3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DFLASH2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, false);
                hparams.f_final_logit_softcapping = 0.0f;
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);
                ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale, false);
                ml.get_key("dflash.block_size",       hparams.dflash_block_size);
                ml.get_key("dflash.conv_kernel_size", hparams.dflash_conv_kernel_size);
                ml.get_key("dflash.conv_group_size",  hparams.dflash_conv_group_size);
                ml.get_key("dflash.selector_rank",    hparams.dflash_selector_rank);
                ml.get_key("dflash.selector_top_k",   hparams.dflash_selector_top_k);
                ml.get_key(LLM_KV_TOKENIZER_MASK_ID,       hparams.dflash_mask_token_id);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,        hparams.causal_attn, false);
                load_dflash_target_layer_ids(
                        ml,
                        LLM_KV(model.arch)(LLM_KV_DFLASH_TARGET_LAYERS),
                        hparams,
                        true);
                for (uint32_t i = 0; i < hparams.dflash_n_target_layers; ++i) {
                    if (hparams.dflash_target_layer_ids[i] == 0) {
                        throw std::runtime_error("dflash2: target_layers must use one-based IDs");
                    }
                }
                hparams.dflash_n_target_features = hparams.n_embd * hparams.dflash_n_target_layers;
                if (hparams.dflash_selector_top_k == 0 ||
                        hparams.n_embd < hparams.dflash_selector_top_k * (hparams.dflash_selector_top_k + 1)) {
                    throw std::runtime_error("dflash2: hidden size is too small for selector top-k lattice");
                }
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer, false);
                hparams.n_layer_kv_from_start = hparams.n_layer;
                hparams.dflash_dsv4 = false;
                model.type = e_model::MODEL_UNKNOWN;
            } break;
        case LLM_ARCH_DFLASH:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_GLM_DSA:
            {
                if (model.arch == LLM_ARCH_DFLASH) {
                    const bool has_dense_signature =
                            ml.get_tensor_meta("blk.0.attn_q.weight") != nullptr;
                    const bool has_dsv4_signature =
                            ml.get_tensor_meta("blk.0.attn_q_a.weight") != nullptr ||
                            ml.get_tensor_meta("blk.0.hc_attn_base.weight") != nullptr;
                    if (has_dense_signature && has_dsv4_signature) {
                        throw std::runtime_error("dflash: ambiguous dense and DSV4 tensor signatures");
                    }
                    if (!has_dense_signature && !has_dsv4_signature) {
                        throw std::runtime_error("dflash: unrecognized or incomplete tensor signature");
                    }
                    hparams.dflash_dsv4 = has_dsv4_signature;
                }
                const bool dflash_dense = model.arch == LLM_ARCH_DFLASH && !hparams.dflash_dsv4;
                if (dflash_dense) {
                    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                    ml.get_key("dflash.block_size", hparams.dflash_block_size);
                    ml.get_key(LLM_KV_TOKENIZER_MASK_ID, hparams.dflash_mask_token_id);
                    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer, false);
                    load_dflash_target_layer_ids(ml, "dflash.target_layers", hparams, true);

                    const ggml_tensor * fc = ml.get_tensor_meta("fc.weight");
                    if (fc == nullptr || fc->ne[0] <= 0 || fc->ne[1] != hparams.n_embd) {
                        throw std::runtime_error("dflash: fc.weight must have shape [n_target_features, embedding_length]");
                    }
                    hparams.dflash_n_target_features = (uint32_t) fc->ne[0];
                    hparams.dflash_backbone_rotary_base = hparams.rope_freq_base_train;
                    hparams.dflash_laguna = false;

                    for (uint32_t i = 0; i < hparams.dflash_n_target_layers; ++i) {
                        if (hparams.dflash_target_layer_ids[i] == 0) {
                            throw std::runtime_error("dflash: target_layers must use one-based IDs");
                        }
                        --hparams.dflash_target_layer_ids[i];
                    }
                    validate_dflash_hparams(hparams, model.arch);

                    hparams.n_layer_kv_from_start = hparams.n_layer;
                    model.type = e_model::MODEL_UNKNOWN;
                    break;
                }
                const bool is_dsv4 = llm_arch_is_dsv4(model.arch) || hparams.dflash_dsv4;
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                // keep the two nextn spellings in step -- see llama-hparams.h
                hparams.sync_nextn_from_nextn_predict_layers();
                if (llm_arch_is_dsv4(model.arch) && hparams.n_layer == 43 && hparams.nextn_predict_layers > 0) {
                    LLAMA_LOG_WARN("===============================================================================================\n");
                    LLAMA_LOG_WARN("Unexpected number of layers (%d) and nextn_predict_layers (%d) for DeepSeek4-Flash\n",
                            hparams.n_layer, hparams.nextn_predict_layers);
                    LLAMA_LOG_WARN("   -> setting nextn_predict_layers to zero\n");
                    LLAMA_LOG_WARN("===============================================================================================\n");
                    hparams.nextn_predict_layers = 0;
                }
                // Probe the first appended predictor block, or n_layer - 1 for base GGUFs.
                const uint32_t dsv4_probe_offset = std::max<uint32_t>(1, hparams.nextn_predict_layers);
                const uint32_t dsv4_probe_layer = hparams.n_layer > dsv4_probe_offset
                    ? hparams.n_layer - dsv4_probe_offset
                    : 0;
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,     hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
                // GLM-DSA lightning-indexer k_norm is a (non-RMS) LayerNorm built via LLM_NORM,
                // which uses hparams.f_norm_eps in ggml_norm(). The GGUF only carries the RMS eps,
                // so f_norm_eps stays 0 and CPU ggml_norm aborts (GGML_ASSERT(eps > 0)). On CUDA
                // the kernel does not assert (eps=0 is numerically tolerable), which is why the
                // CPU attention path was never exercised. Mirror the RMS eps so the indexer
                // LayerNorm gets a valid epsilon on all backends. (HF hardcodes 1e-6 for this
                // LayerNorm, but 1e-6 vs 1e-5 is within 4-chunk PPL noise here, so keep the mirror.)
                if (hparams.f_norm_eps <= 0.0f) hparams.f_norm_eps = hparams.f_norm_rms_eps;
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false, true);  // mrope tolerant: DEC-S69-MROPE-SECTIONS-TOLERANT

                // MoE parameters
                ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
                ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

                if (is_dsv4) {
                    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp,   hparams.n_layer);
                    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_clamp_shexp, hparams.n_layer, 0)) {
                        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
                    }
                }

                // Shared latent-attention ranks used by common hparams and
                // cache helpers. DSV4 has its own CSA/HCA execution graph;
                // this does not select the DeepSeek V3 MLA path.
                ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv, false);
                if (is_dsv4 && hparams.n_lora_kv == 0) {
                    const uint32_t probe_layer = dsv4_probe_layer;
                    if (auto * kv_norm = ml.get_tensor_meta(format("blk.%u.attn_kv_a_norm.weight", probe_layer).c_str())) {
                        hparams.n_lora_kv = (uint32_t) kv_norm->ne[0];
                    } else if (auto * kv = ml.get_tensor_meta(format("blk.%u.attn_kv.weight", probe_layer).c_str())) {
                        const int64_t kv_inner = kv->ne[0] == hparams.n_embd ? kv->ne[1] : kv->ne[0];
                        hparams.n_lora_kv = (uint32_t) kv_inner;
                    } else {
                        auto * kv_a = ml.get_tensor_meta(format("blk.%u.attn_kv_latent.weight", probe_layer).c_str());
                        bool subtract_rope = false;
                        if (kv_a == nullptr) {
                            kv_a = ml.require_tensor_meta(format("blk.%u.attn_kv_a_mqa.weight", probe_layer).c_str());
                            subtract_rope = true;
                        }

                        const int64_t kv_a_inner = kv_a->ne[0] == hparams.n_embd ? kv_a->ne[1] : kv_a->ne[0];
                        if (subtract_rope) {
                            if (kv_a_inner <= hparams.n_rot) {
                                throw std::runtime_error(format(
                                    "%s: unable to infer %s from blk.0.attn_kv_a_mqa.weight shape [%lld, %lld]",
                                    __func__,
                                    ml.llm_kv(LLM_KV_ATTENTION_KV_LORA_RANK).c_str(),
                                    (long long) kv_a->ne[0],
                                    (long long) kv_a->ne[1]));
                            }
                            hparams.n_lora_kv = (uint32_t) (kv_a_inner - hparams.n_rot);
                        } else {
                            hparams.n_lora_kv = (uint32_t) kv_a_inner;
                        }
                    }
                }
                //ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
                //ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);

                // DSA parameters
                ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

                if (is_dsv4) {
                    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
                    hparams.rope_freq_scale_train_swa = 1.0f;
                    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer, false) && hparams.n_swa > 0) {
                        std::fill(hparams.is_swa_impl.begin(), hparams.is_swa_impl.end(), true);
                    }
                    // upstream pairs the all-layers pattern with STANDARD (SOURCE-READ upstream/llama.cpp/src/models/deepseek4.cpp:67-68, MIT);
                    // here only when the pattern actually marked a layer, so a DSV4 file without a window keeps swa_type NONE
                    if (hparams.is_swa_any()) {
                        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                    }
                    // DSV4 vision: in a non-causal (image) span the window only clips older tokens
                    hparams.swa_full_non_causal = true;

                    const auto * hc_head_base = ml.get_tensor_meta("hc_head_base");
                    const uint32_t probe_layer = dsv4_probe_layer;
                    const auto * wo_a_0 = ml.get_tensor_meta(format("blk.%u.attn_output_a.weight", probe_layer).c_str());
                    const auto * wo_b_0 = ml.get_tensor_meta(format("blk.%u.attn_output_b.weight", probe_layer).c_str());

                    if (!ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT, hparams.dsv4_o_group_count, false) && wo_a_0 != nullptr) {
                        GGML_ASSERT(wo_a_0->ne[0] > 0);
                        hparams.dsv4_o_group_count = (uint32_t) ((hparams.n_head() * hparams.n_embd_head_k(0)) / wo_a_0->ne[0]);
                    }
                    if (!ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK, hparams.dsv4_o_lora_rank, false) && wo_b_0 != nullptr) {
                        GGML_ASSERT(hparams.dsv4_o_group_count > 0);
                        hparams.dsv4_o_lora_rank = (uint32_t) (wo_b_0->ne[0] / hparams.dsv4_o_group_count);
                    }
                    if (!ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, hparams.dsv4_compress_rope_base, false)) {
                        hparams.dsv4_compress_rope_base = hparams.rope_freq_base_train_swa != 0.0f
                            ? hparams.rope_freq_base_train_swa
                            : hparams.rope_freq_base_train;
                    }
                    if (!ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false)) {
                        if (hc_head_base != nullptr) {
                            hparams.dsv4_hc_mult = (uint32_t) hc_head_base->ne[0];
                        } else if (wo_a_0 != nullptr) {
                            hparams.dsv4_hc_mult = (uint32_t) (wo_a_0->ne[1] / hparams.n_embd);
                        }
                    }
                    // Base GGUFs lack companion output width, derive it from target HC width.
                    if (hparams.n_embd_out == hparams.n_embd && hparams.dsv4_hc_mult > 1) {
                        hparams.n_embd_out = hparams.n_embd * hparams.dsv4_hc_mult;
                    }
                    if (!ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters, false)) {
                        hparams.dsv4_hc_sinkhorn_iters = 3;
                    }
                    if (!ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON, hparams.dsv4_hc_eps, false)) {
                        hparams.dsv4_hc_eps = hparams.f_norm_rms_eps;
                    }
                    ml.get_key(LLM_KV_HASH_LAYER_COUNT, hparams.dsv4_hash_layer_count, false);

                    if (model.arch == LLM_ARCH_DEEPSEEK41) {
                        ml.get_arr_n(LLM_KV_ENGRAM_LAYER_IDS, hparams.engram_n_layer);
                        if (hparams.engram_n_layer == 0 || hparams.engram_n_layer > LLAMA_MAX_LAYERS) {
                            throw std::runtime_error(format("DeepSeek-V4.1 engram layer count %u is out of range", hparams.engram_n_layer));
                        }
                        {
                            std::vector<uint32_t> ids;
                            ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_LAYER_IDS), ids);
                            std::copy_n(ids.begin(), std::min<size_t>(ids.size(), LLAMA_MAX_LAYERS), hparams.engram_layer_ids.begin());
                        }
                        ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT,     hparams.engram_n_head);
                        ml.get_key(LLM_KV_ENGRAM_KEY_LENGTH,     hparams.engram_key_length);
                        ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, hparams.engram_max_ngram_size);
                        ml.get_key(LLM_KV_ENGRAM_PAD_ID,         model.engram_pad_id);
                        if (hparams.engram_n_head == 0 || hparams.engram_max_ngram_size < 2) {
                            throw std::runtime_error("DeepSeek-V4.1 engram needs at least one head and a 2-gram");
                        }
                        for (uint32_t e = 0; e < hparams.engram_n_layer; ++e) {
                            if (hparams.engram_layer_ids[e] >= hparams.n_layer) {
                                throw std::runtime_error(format("engram layer %u is out of range", hparams.engram_layer_ids[e]));
                            }
                        }
                        ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_MULTIPLIERS), model.engram_multipliers);
                        ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_PRIMES),      model.engram_primes);
                        ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_OFFSETS),     model.engram_offsets);
                        ml.get_arr(ml.llm_kv(LLM_KV_ENGRAM_TOKEN_MAP),   model.engram_token_map);
                        const size_t n_bucket = (size_t) (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
                        if (model.engram_multipliers.size() != (size_t) hparams.engram_n_layer * hparams.engram_max_ngram_size) {
                            throw std::runtime_error("engram multiplier count does not match layers * ngram size");
                        }
                        if (model.engram_primes.size() != (size_t) hparams.engram_n_layer * n_bucket ||
                            model.engram_offsets.size() != model.engram_primes.size()) {
                            throw std::runtime_error("engram prime or offset count does not match layers * buckets");
                        }
                        for (uint64_t p : model.engram_primes) {
                            if (p == 0) throw std::runtime_error("engram prime of zero would divide by zero in the hash");
                        }
                    }

                    uint32_t n_compress_ratios = 0;
                    if (ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios, false)) {
                        if (n_compress_ratios < hparams.n_layer) {
                            throw std::runtime_error("DeepSeek-V4 compress_ratios is shorter than block_count");
                        }
                        std::vector<uint32_t> compress_ratios;
                        ml.get_arr(ml.llm_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS), compress_ratios);
                        std::copy_n(compress_ratios.begin(), hparams.n_layer, hparams.dsv4_compress_ratios.begin());
                    } else {
                        for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                            const bool has_attn_compress =
                                ml.get_tensor_meta(format("blk.%u.attn_compress_kv.weight",   il).c_str()) != nullptr ||
                                ml.get_tensor_meta(format("blk.%u.attn_compressor_kv.weight", il).c_str()) != nullptr;
                            const bool has_indexer =
                                ml.get_tensor_meta(format("blk.%u.indexer.attn_q_b.weight",       il).c_str()) != nullptr ||
                                ml.get_tensor_meta(format("blk.%u.indexer.compress_kv.weight",    il).c_str()) != nullptr ||
                                ml.get_tensor_meta(format("blk.%u.indexer_compressor_kv.weight",  il).c_str()) != nullptr;

                            if (has_indexer && !has_attn_compress) {
                                throw std::runtime_error(format("DeepSeek-V4 layer %u has indexer tensors without attention compressor tensors", il));
                            }

                            hparams.dsv4_compress_ratios[il] = has_indexer ? 4 : (has_attn_compress ? 128 : 0);
                        }
                    }

                    if (model.arch == LLM_ARCH_DEEPSEEK41) {
                        // a source carries a compressor, a key owner indexer.attn_k, an index source
                        // indexer.attn_q_b: walk the layers once and record who reads from whom
                        hparams.dsv4_shared_streams = true;
                        hparams.dsv41_kv_source.fill(-1);
                        hparams.dsv41_index_key_source.fill(-1);
                        hparams.dsv41_topk_source.fill(-1);
                        int32_t last_kv = -1, last_key = -1, last_idx = -1;
                        uint32_t csa_ratio = 0, hca_ratio = 0;
                        for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                            const bool has_comp  = ml.get_tensor_meta(format("blk.%u.attn_compressor_kv.weight", il).c_str()) != nullptr;
                            const bool has_gate  = ml.get_tensor_meta(format("blk.%u.attn_compressor_gate.weight", il).c_str()) != nullptr;
                            const bool has_idx_k = ml.get_tensor_meta(format("blk.%u.indexer.attn_k.weight", il).c_str()) != nullptr;
                            const bool has_idx_q = ml.get_tensor_meta(format("blk.%u.indexer.attn_q_b.weight", il).c_str()) != nullptr;
                            if (has_comp)  last_kv  = (int32_t) il;
                            if (has_idx_k) last_key = (int32_t) il;
                            if (has_idx_q) last_idx = (int32_t) il;
                            const uint32_t r = hparams.dsv4_compress_ratios[il];
                            if (r == 0) continue;
                            if (last_kv < 0 || last_key < 0 || last_idx < 0) {
                                throw std::runtime_error(format("DeepSeek-V4.1 layer %u reads a compressed stream before any layer publishes one", il));
                            }
                            if (r != hparams.dsv4_compress_ratios[last_kv]) {
                                throw std::runtime_error(format("DeepSeek-V4.1 layer %u compresses at ratio %u but reads layer %d, compressed at %u",
                                            il, r, last_kv, hparams.dsv4_compress_ratios[last_kv]));
                            }
                            if (has_comp && !has_gate && r != 1) {
                                throw std::runtime_error(format("DeepSeek-V4.1 layer %u pools %u tokens per row but has no pooling gate", il, r));
                            }
                            hparams.dsv41_kv_source[il]        = last_kv;
                            hparams.dsv41_index_key_source[il] = last_key;
                            hparams.dsv41_topk_source[il]      = last_idx;
                            if (r == csa_ratio || r == hca_ratio) continue;
                            if      (csa_ratio == 0) csa_ratio = r;
                            else if (hca_ratio == 0) hca_ratio = r;
                            else throw std::runtime_error("DeepSeek-V4.1 supports at most two compression ratios");
                        }
                        // a file with no compressed layers is pure sliding window attention; the
                        // ratios only have to stay non-zero for the size arithmetic
                        if (csa_ratio == 0) csa_ratio = 1;
                        if (hca_ratio == 0) hca_ratio = csa_ratio;
                        // V4 splits the two streams by role -- the indexed one at CSA_RATIO, the
                        // other at HCA_RATIO. V4.1 does not: both of its streams carry index
                        // sources (indexer.attn_q_b on layers 2, 8, 14 at ratio 2 and on 20, 24,
                        // 28, 32, 36 at ratio 1 in the released Flash file), so these two hold the
                        // file's compression segments in layer order and nothing reads them as
                        // roles -- the cache arithmetic and the graph both go through the ratio.
                        hparams.dsv4_csa_ratio = csa_ratio;
                        hparams.dsv4_hca_ratio = hca_ratio;
                        hparams.dsv4_csa_overlap = false;
                        hparams.dsv4_hca_overlap = false;
                        hparams.dsv4_hc_lag = true;
                        hparams.dsv4_q_head_norm = false;
                        LLAMA_LOG_INFO("%s: DeepSeek-V4.1 compressed streams: csa ratio %u, hca ratio %u, shared from source layers\n", __func__, csa_ratio, hca_ratio);
                    }
                    if (hparams.is_v41_separate()) {
                        {
                            uint32_t src = 0;
                            if (ml.get_key(LLM_KV_CANDIDATE_SOURCE_LAYER, src, false)) {
                                hparams.dsv4_candidate_source_layer = (int32_t) src;
                            }
                        }
                        ml.get_key(LLM_KV_CANDIDATE_BLOCK_SIZE,  hparams.dsv4_candidate_block_size,  false);
                        ml.get_key(LLM_KV_CANDIDATE_TOPK_BLOCKS, hparams.dsv4_candidate_topk_blocks, false);
                        if (hparams.dsv4_candidate_source_layer < 0) { hparams.dsv4_candidate_source_layer = 20; }
                        if (hparams.dsv4_candidate_block_size == 0)  { hparams.dsv4_candidate_block_size  = 8; }
                        if (hparams.dsv4_candidate_topk_blocks == 0) { hparams.dsv4_candidate_topk_blocks = 2048; }
                        LLAMA_LOG_INFO("%s: DSV4.1 hierarchical indexer: candidate source layer = %d, block size = %u, top-k blocks = %u\n",
                                __func__, hparams.dsv4_candidate_source_layer,
                                hparams.dsv4_candidate_block_size, hparams.dsv4_candidate_topk_blocks);
                    }
                    if (hparams.dsv4_hc_mult == 0) {
                        throw std::runtime_error("DeepSeek-V4 hyper_connection.count is missing and could not be inferred");
                    }
                    if (hparams.dsv4_o_group_count == 0 || hparams.dsv4_o_lora_rank == 0) {
                        throw std::runtime_error("DeepSeek-V4 output projection metadata is missing and could not be inferred");
                    }
                    if (wo_b_0 != nullptr && (int64_t) hparams.dsv4_o_group_count * hparams.dsv4_o_lora_rank != wo_b_0->ne[0]) {
                        throw std::runtime_error("DeepSeek-V4 inferred output_group_count/output_lora_rank does not match attn_output_b shape");
                    }
                    if (wo_a_0 != nullptr && (int64_t) hparams.dsv4_o_group_count * wo_a_0->ne[0] != (int64_t) hparams.n_head() * hparams.n_embd_head_k(0)) {
                        throw std::runtime_error("DeepSeek-V4 inferred output_group_count does not match attn_output_a shape");
                    }
                }

                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.is_indexer_full_impl[il] = (il <= 1) || (il % 4 == 2);
                }

                // Expert gating function (GLM-4.5 uses sigmoid)
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }

                if (is_dsv4 &&
                    hparams.expert_gating_func != LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
                    throw std::runtime_error("DeepSeek-V4 loader currently expects sqrtsoftplus MoE scoring");
                }

                if (is_dsv4) {
                    hparams.n_layer_kv_from_start = hparams.n_layer;
                } else {
                    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                    // keep the two nextn spellings in step -- see llama-hparams.h
                    hparams.sync_nextn_from_nextn_predict_layers();
                    hparams.n_layer_kv_from_start = model.mtp
                        ? hparams.n_layer
                        : hparams.n_layer - hparams.nextn_predict_layers;
                }

                switch (hparams.n_layer) {
                    case 61: model.type = MODEL_290B; break;
                    case 79: model.type = MODEL_744B_A40B; break;
                    default: model.type = MODEL_UNKNOWN;
                }
                if (!is_dsv4 && hparams.n_head_kv() == 1) {
                    int n_nead_kv = hparams.n_gqa();
                    if (n_nead_kv%4 != 0 || hparams.n_embd_head_k_full != 576 || hparams.n_embd_head_v_full != 512 ||
                        hparams.n_rot != 64) {
                        LLAMA_LOG_ERROR("==========================================================================\n");
                        LLAMA_LOG_ERROR("Detected incompatible DeepSeek model without a known way to fix it.\n");
                        LLAMA_LOG_ERROR("Sorry, unknown model => cannot fix it => bailing out\n");
                        LLAMA_LOG_ERROR("==========================================================================\n");
                        GGML_ABORT("Fatal error");
                    }
                    LLAMA_LOG_INFO("================= Adjusted mainline llama.cpp MLA tensors to ik_llama.cpp\n");
                    for (auto& item : hparams.n_head_kv_arr) item = n_nead_kv;
                    hparams.n_embd_head_k_full = 192;
                    hparams.n_embd_head_v_full = 128;
                    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_full);
                    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_full);
                }

                if (model.arch == LLM_ARCH_DFLASH) {
                    hparams.dflash_dsv4 = hparams.dsv4_hc_mult > 0;
                    if (!hparams.dflash_dsv4) {
                        throw std::runtime_error("dflash: hyper_connection.count is required for the official DSV4 schema");
                    }
                    // a V4.1 draft lags its hyper-connection mixes and collapses the output with the last
                    // FFN's mix, as the body does; the absence of output_hc_base.weight is the signature
                    hparams.dflash_dsv41 = ml.get_tensor_meta("output_hc_base.weight") == nullptr;
                    hparams.dflash_block_bidir = hparams.dflash_dsv41;
                    LLAMA_LOG_INFO("%s: DSV4 draft flavor = %s\n", __func__,
                            hparams.dflash_dsv41 ? "V4.1 (lagged hyper-connections, no output head)" : "V4");

                    ml.get_key("dflash.block_size", hparams.dflash_block_size, true);
                    ml.get_key(LLM_KV_TOKENIZER_MASK_ID, hparams.dflash_mask_token_id, true);
                    load_dflash_target_layer_ids(
                            ml,
                            LLM_KV(model.arch)(LLM_KV_DFLASH_TARGET_LAYERS),
                            hparams,
                            true);

                    for (uint32_t i = 0; i < hparams.dflash_n_target_layers; ++i) {
                        if (hparams.dflash_target_layer_ids[i] == 0) {
                            throw std::runtime_error("dflash: target_layers must use one-based IDs");
                        }
                        --hparams.dflash_target_layer_ids[i];
                    }

                    hparams.dflash_n_target_features = hparams.n_embd * hparams.dflash_n_target_layers;
                    hparams.dflash_laguna = false;
                    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                        if (hparams.dsv4_compress_ratios[i] != 0) {
                            throw std::runtime_error("dflash: DSV4 draft requires uncompressed stages");
                        }
                    }
                    validate_dflash_hparams(hparams, model.arch);
                }
            } break;
        case LLM_ARCH_GLM5NEXT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                // The DSA lightning-indexer k_norm is a plain (non-RMS) LayerNorm built via
                // LLM_NORM, which uses hparams.f_norm_eps in ggml_norm(). The GGUF only carries
                // the RMS eps, so mirror it as the DEEPSEEK4/GLM_DSA blocks do.
                if (hparams.f_norm_eps <= 0.0f) {
                    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps, false);
                    if (hparams.f_norm_eps <= 0.0f) {
                        hparams.f_norm_eps = hparams.f_norm_rms_eps;
                    }
                }

                // The NextN/MTP block is appended after the trunk; trunk-only GGUFs still load
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
                // believe the tensors: a stale count hides the last real layer
                if (hparams.nextn_predict_layers > 0) {
                    const std::string probe = LLM_TN(LLM_ARCH_GLM5NEXT)(LLM_TENSOR_NEXTN_EH_PROJ, "weight",
                            hparams.n_layer - hparams.nextn_predict_layers);
                    if (ml.get_tensor_meta(probe.c_str()) == nullptr) {
                        hparams.nextn_predict_layers = 0;
                    }
                }

                // MLA. GLM-5.3-Flash is nope-only (rope.dimension_count == 0), so n_rot stays 0 and
                // the whole q head is the NoPE part
                ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
                ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_full);
                ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_full);

                // MoE with shared experts; GLM-5.3-Flash uses sigmoid gating
                ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
                ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared, false);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                if (hparams.n_ff_shexp == 0 && hparams.n_expert_shared > 0) {
                    hparams.n_ff_shexp = hparams.n_ff_exp;
                }
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }
                // clamped SwiGLU (gate clamped before the SiLU, deepseek4 semantics)
                ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp,        hparams.n_layer, false);
                if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer, false)) {
                    hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
                }

                // DSA k-pool indexer. Absent in a GGUF without indexer weights, which then
                // runs dense MLA; indexer_block_size (kpool) is the tokens-per-compressed-key cell
                ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head,    false);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size, false);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k,     false);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL,      hparams.indexer_block_size, false);
                if (hparams.indexer_head_size > 0) {
                    GGML_ASSERT(hparams.indexer_n_head > 0 && "GLM5NEXT indexer key length without indexer head count");
                    if (hparams.indexer_block_size > 0 && hparams.indexer_top_k > 0 &&
                            hparams.indexer_top_k % hparams.indexer_block_size != 0) {
                        LLAMA_LOG_WARN("%s: indexer.top_k (%u) is not a multiple of indexer.kpool (%u); top-k will be clamped to whole pools\n",
                                __func__, hparams.indexer_top_k, hparams.indexer_block_size);
                    }
                }

                // hyper-connections (mHC) wrap every trunk attention/FFN block
                ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult);
                ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
                if (!ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON, hparams.dsv4_hc_eps, false)) {
                    hparams.dsv4_hc_eps = hparams.f_norm_rms_eps;
                }
                if (hparams.dsv4_hc_mult == 0) {
                    throw std::runtime_error("GLM5NEXT: hyper_connection.count is required");
                }
                if (hparams.dsv4_hc_sinkhorn_iters == 0) {
                    throw std::runtime_error("GLM5NEXT: hyper_connection.sinkhorn_iterations is required");
                }

                // KDA (kimi-k3 low-rank parameterization on the KDA layers)
                ml.get_key(LLM_KV_SSM_CONV_KERNEL, hparams.ssm_d_conv);
                ml.get_key(LLM_KV_KDA_HEAD_DIM, hparams.ssm_d_state, false);
                if (hparams.ssm_d_state == 0) {
                    // older converters store the KDA head dim as ssm.state_size
                    ml.get_key(LLM_KV_SSM_STATE_SIZE, hparams.ssm_d_state);
                }
                ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND, hparams.kda_gate_lower_bound, false);
                if (hparams.kda_gate_lower_bound == 0.0f) {
                    hparams.kda_gate_lower_bound = -5.0f;
                }
                GGML_ASSERT(hparams.kda_gate_lower_bound < 0.0f && "GLM5NEXT: only the bounded sigmoid gate is implemented");
                hparams.ssm_dt_rank = hparams.n_head();
                // no ssm.inner_size key in the GGUF: d_inner = head_dim * num_heads
                hparams.ssm_d_inner = hparams.ssm_d_state * hparams.ssm_dt_rank;
                // ssm_n_group (num KDA heads) is not in the GGUF; derive from n_head like bailingmoe3
                if (!ml.get_key(LLM_KV_SSM_GROUP_COUNT, hparams.ssm_n_group, false)) {
                    hparams.ssm_n_group = hparams.n_head();
                } else if (hparams.ssm_n_group == 0 || hparams.ssm_n_group > hparams.ssm_dt_rank ||
                           hparams.ssm_dt_rank % hparams.ssm_n_group != 0) {
                    // KDA keeps one state per head (upstream glm5-next.cpp never reads ssm.group_count); a count that does not
                    // divide the head count cannot describe it, and the state layout divides by dt_rank / n_group
                    LLAMA_LOG_WARN("%s: GLM5NEXT: ssm.group_count = %u does not divide the %u KDA heads; using %u\n", __func__,
                                   hparams.ssm_n_group, hparams.ssm_dt_rank, hparams.n_head());
                    hparams.ssm_n_group = hparams.n_head();
                }

                // per-layer head_count_kv array: 0 marks the KDA (linear-attention) layers, 1 the MLA
                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.is_recr_impl[il] = hparams.n_head_kv_arr[il] == 0;
                }
                // unlike GLM-5.2 IndexShare, every MLA layer computes its own DSA top-k
                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.is_indexer_full_impl[il] = !hparams.is_recr_impl[il];
                }

                hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;

                switch (hparams.n_layer - hparams.nextn_predict_layers) {
                    case 45: model.type = e_model::MODEL_312B_A17B; break; // GLM-5.3-Flash
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MUSE_GLIMMER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);
                ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);

                hparams.rope_freq_base_train_swa = hparams.rope_freq_base_train;
                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
                // SOURCE-READ upstream/llama.cpp/src/models/muse-glimmer.cpp:12 (MIT): the pattern below is paired with STANDARD
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

                if (uint32_t swa_period; ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false) && swa_period > 0) {
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        hparams.is_swa_impl[il] = (il % swa_period < swa_period - 1);
                    }
                } else if (!ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer)) {
                    LLAMA_LOG_WARN("================================ No attention.sliding_window_pattern key found! Assuming a period of 4\n");
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        hparams.is_swa_impl[il] = (il % 4 < 3);
                    }
                }

                switch (hparams.n_layer - hparams.nextn_predict_layers) {
                    case 52: model.type = e_model::MODEL_30B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

            } break;
        default:
            // An architecture with no case above whose model object is a src/models class (llama_model_create_for_arch): its own loader reads the arch-specific keys.
            // The common keys it relies on were all read at the top of this function, as upstream's llama_model::load_hparams does before calling it.
            if (auto * mb = dynamic_cast<llama_model_base *>(&model)) {
                mb->ml = &ml;
                // Upstream's struct default for the SWA rope (llama-hparams.h: 10000 / 1.0), NOT the full-attention base the common block above copies in for the
                // legacy archs. A GGUF without rope.freq_base_swa (gemma-3n-E2B) otherwise runs its sliding layers at base 1e6: KLD 0.357 vs stock, measured 2026-10-04.
                hparams.rope_freq_base_train_swa  = 10000.0f;
                hparams.rope_freq_scale_train_swa = 1.0f;
                // Same reason, three more members whose fork struct default is llama4's value, not upstream's (found by diffing the two llama_hparams default sets,
                // _tca_hparams_defaults_diff_1004.py): the class deepseek2/mistral3 graphs branch on f_attn_temp_scale != 0, so a GGUF without the key got attention
                // temperature scaling it must not have. llama4 sets all three itself (models/llama4.cpp).
                hparams.f_attn_temp_scale       = 0.0f;
                hparams.n_attn_temp_floor_scale = 0;
                hparams.use_kq_norm             = false;
                mb->load_arch_hparams(ml);
                switch (model.arch) {
                    case LLM_ARCH_MAMBA2:
                    case LLM_ARCH_FALCON_H1:
                    case LLM_ARCH_NEMOTRON_H:
                    case LLM_ARCH_NEMOTRON_H_MOE:
                        hparams.ssm_split_state = true;
                        break;
                    default:
                        break;
                }
            }
            break;
    }

    // Two fields both say "this model has sliding-window layers": swa_type (read by create_memory to pick
    // llama_kv_cache_iswa) and is_swa_impl (filled by set_swa_pattern or the SWA-pattern key). An arch that
    // fills one and not the other reaches GGML_ASSERT(!hparams.is_swa_any()) deep in create_memory with no
    // hint of which loader is wrong -- GEMMA3 (449ff70d6) and GEMMA4 both shipped in that state. WARN here,
    // naming the arch, so the abort that may follow is attributable. Not a throw: 13 arch cases fill
    // is_swa_impl without swa_type (temp/_swa_0928_scan.py, 2026-09-28) and some may leave create_memory
    // through an earlier arm and load today; each gets swa_type only when a model of it is verified.
    // Row: S69-SWA-PATTERN-SET-WITHOUT-SWA-TYPE-LLAMA4-COHERE2-2026-09-14-01.
    if (hparams.swa_type == LLAMA_SWA_TYPE_NONE && hparams.is_swa_any()) {
        LLAMA_LOG_WARN("%s: arch '%s' marks sliding-window layers (is_swa_impl) but leaves hparams.swa_type NONE; "
                "llama_model::create_memory aborts on GGML_ASSERT(!hparams.is_swa_any()) if it reaches its "
                "non-SWA arm -- set swa_type in this arch's case (see upstream src/models/<arch>.cpp)\n",
                __func__, llama_model_arch_name(model.arch));
    }

    model.ftype = ml.ftype;

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_rope_type(&model);
}

// ================================================================================
// ADOPTED 2026-09-08 from 279b4bfaf^2:src/llama-hparams.cpp -- six methods, bodies verbatim
// except where this fork's spelling differs, which is marked at the line.
//
// These are not accessors: four of them COMPUTE a value from other hparams. Writing a body from
// the method name would be inventing a mechanism, so each is upstream's own text.
// ================================================================================

bool llama_hparams::is_mla() const {
    GGML_ASSERT((n_embd_head_k_mla_impl == 0 && n_embd_head_v_mla_impl == 0) ||
                (n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0));

    return n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0;
}

uint32_t llama_hparams::n_embd_head_k_mla() const {
    return is_mla() ? n_embd_head_k_mla_impl : n_embd_head_k();
}

uint32_t llama_hparams::n_embd_head_v_mla() const {
    return is_mla() ? n_embd_head_v_mla_impl : n_embd_head_v();
}

bool llama_hparams::has_rope(uint32_t il) const {
    // the router layer stores adapter routing signal, not positional info,
    // so it must not be RoPE-shifted
    if (router_layer >= 0 && (int32_t) il == router_layer) {
        return false;
    }

    // Upstream reads n_layer_all here; this fork spells the total layer count n_layer. Same fact,
    // proven by both structs' own definitions -- upstream's n_layer_all IS the total and its
    // n_layer() METHOD is the effective count, which this fork spells n_layer_eff().
    if (il < n_layer) {
        return rope_pattern[il] != 0;
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer: %u)\n", __func__, il, n_layer);
}

uint32_t llama_hparams::n_embd_inp_enc() const {
    return n_embd_inp_enc_impl > 0 ? n_embd_inp_enc_impl : n_embd_inp();
}

bool llama_hparams::use_mrope() const {
    return rope_sections[0] > 0 && rope_sections[1] > 0;
}

// ik #2878 (97a5bce89): the DeepSeek-V4.1 separate builder is the default; V41_SEPARATE=0 selects the in-threaded one
bool llama_hparams::is_v41_separate() {
    static const bool is_separate = [] () -> bool {
        if (auto env = std::getenv("V41_SEPARATE"); env != nullptr && env[0] == '0') {
            return false;
        }
        return true;
    }();
    return is_separate;
}
