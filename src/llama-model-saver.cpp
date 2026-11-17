#include "llama-model-saver.h"

#include "ggml.h"
#include "gguf.h"

#include "llama-arch.h"
#include "llama.h"
#include "llama-hparams.h"
#include "llama-model.h"
#include "llama-vocab.h"
#include "models/models.h"

#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

bool llama_model_saver_supports_arch(llm_arch arch) {
    switch (arch) {
        // llama.cpp #29042 (MIT) took plamo3 gemma3 cohere2 cohere2moe olmo2 exaone-moe afmoe mimo2 muse-glimmer mellum laguna
        // granite-swa dots3note off this list (SWA pattern + MLA-SWA keys now written). saverrest-1019 took the last six off, each
        // with test-llama-archs -a <arch> failing with the gate opened and passing after the fix (same build tree):
        //   bitnet, step35  the gate alone held them; they round-trip bit-exact with no saver change
        //   gemma3n         reload stopped at "tensor 'altup_proj.weight' not found": model-level AltUp + per-layer token embedding now written
        //   glm-dsa         reload stopped at "expected 46, got 44": the load-computed wkv_b (llm_prepare_mla) is no longer written
        //   t5              reload stopped at "decoder_start_token_id has wrong type i32 but expected type u32": written as u32
        //   apertus         reload stopped at "xielu.alpha_n has wrong array length; expected 2, got 512": written per layer
        // llama.cpp master still lists GEMMA3N BITNET T5 APERTUS STEP35 (src/llama-model-saver.cpp, read 2026-10-19); it has no fix to port.
        // upstream (merge 2026-11-04) also lists GEMMA3N BITNET T5 APERTUS STEP35: this fork round-trips them (above). Its new entries:
        case LLM_ARCH_SENSENOVA_U1:
        case LLM_ARCH_CLEF: // the head tensors are not saved
        case LLM_ARCH_VLLM_DECISION2:
            return false;
        default:
            return true;
    }
}

llama_model_saver::llama_model_saver(const struct llama_model * model) :
        gguf_ctx(gguf_init_empty()), gguf_ctx_owned(true), model(model), llm_kv(model->arch) {
    GGML_ASSERT(llama_model_saver_supports_arch(model->arch));
}

llama_model_saver::llama_model_saver(enum llm_arch arch, struct gguf_context * gguf_ctx) :
        gguf_ctx(gguf_ctx == nullptr ? gguf_init_empty() : gguf_ctx), gguf_ctx_owned(gguf_ctx == nullptr), model(nullptr), llm_kv(arch) {}

llama_model_saver::~llama_model_saver() {
    if (gguf_ctx_owned) {
        gguf_free(gguf_ctx);
    }
}

void llama_model_saver::add_kv(const enum llm_kv key, const uint32_t value) {
    gguf_set_val_u32(gguf_ctx, llm_kv(key).c_str(), value);
}

void llama_model_saver::add_kv(const enum llm_kv key, const int32_t value) {
    gguf_set_val_i32(gguf_ctx, llm_kv(key).c_str(), value);
}

void llama_model_saver::add_kv(const enum llm_kv key, const uint64_t value) {
    gguf_set_val_u64(gguf_ctx, llm_kv(key).c_str(), value);
}

void llama_model_saver::add_kv(const enum llm_kv key, const float value) {
    gguf_set_val_f32(gguf_ctx, llm_kv(key).c_str(), value);
}

void llama_model_saver::add_kv(const enum llm_kv key, const bool value) {
    gguf_set_val_bool(gguf_ctx, llm_kv(key).c_str(), value);
}

void llama_model_saver::add_kv(const enum llm_kv key, const char * value) {
    gguf_set_val_str(gguf_ctx, llm_kv(key).c_str(), value);
}

[[noreturn]]
void llama_model_saver::add_kv(const enum llm_kv key, const char value) {
    GGML_UNUSED(key);
    GGML_UNUSED(value);
    GGML_ABORT("fatal error"); // this should never be called, only needed to make the template below compile
}

template <typename Container>
void llama_model_saver::add_kv(const enum llm_kv key, const Container & value, const bool per_layer) {
    GGML_ASSERT(model != nullptr || !per_layer);
    const size_t n_values = per_layer ? size_t(model->hparams.n_layer_eff()) : value.size();
    GGML_ASSERT(n_values <= value.size());

    if (n_values == 0) {
        return;
    }

    if (per_layer) {
        bool all_values_the_same = true;
        for (size_t i = 1; i < n_values; ++i) {
            if (value[i] != value[0]) {
                all_values_the_same = false;
                break;
            }
        }
        if (all_values_the_same) {
            add_kv(key, value[0]);
            return;
        }
    }

    if (std::is_same<typename Container::value_type, uint8_t>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_UINT8, value.data(), n_values);
    } else if (std::is_same<typename Container::value_type, int8_t>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_INT8, value.data(), n_values);
    } else if (std::is_same<typename Container::value_type, uint32_t>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_UINT32, value.data(), n_values);
    } else if (std::is_same<typename Container::value_type, bool>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_BOOL, value.data(), n_values);
    } else if (std::is_same<typename Container::value_type, int32_t>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_INT32, value.data(), n_values);
    } else if (std::is_same<typename Container::value_type, uint64_t>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_UINT64, value.data(), n_values);
    } else if (std::is_same<typename Container::value_type, float>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_FLOAT32, value.data(), n_values);
    } else if (std::is_same<Container, std::string>::value) {
        gguf_set_val_str(gguf_ctx, llm_kv(key).c_str(), reinterpret_cast<const char *>(value.data()));
    } else {
        GGML_ABORT("fatal error");
    }
}
// instantiate for external usage:
template void llama_model_saver::add_kv<std::vector<uint32_t>>(const enum llm_kv, const std::vector<uint32_t> &, const bool);
template void llama_model_saver::add_kv<std::vector<float>>(const enum llm_kv, const std::vector<float> &, const bool);
template void llama_model_saver::add_kv<std::vector<uint64_t>>(const enum llm_kv, const std::vector<uint64_t> &, const bool);
template void llama_model_saver::add_kv<std::vector<int32_t>>(const enum llm_kv, const std::vector<int32_t> &, const bool);

void llama_model_saver::add_kv(const enum llm_kv key, const std::vector<std::string> & value) {
    std::vector<const char *> tmp(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        tmp[i] = value[i].c_str();
    }
    gguf_set_arr_str(gguf_ctx, llm_kv(key).c_str(), tmp.data(), tmp.size());
}

void llama_model_saver::add_tensor(const struct ggml_tensor * tensor) {
    if (!tensor) {
        return;
    }
    // "__missing__" is the name llama_model::tn() hands out for a tensor the architecture does not map (llama-model.cpp:2707-2737); a metadata-only
    // load creates a placeholder under it, and a graph can use it numerically (gemma does: its reload only matches if the ONE placeholder is saved).
    // So the first one is saved like any tensor; a second and later one cannot be (same name) and must not trip the duplicate-name assert below
    // (MEASURED 2026-10-04: granite, granitemoe, glm4, refact, plamo and k2-horizon aborted here on a second '__missing__'; an unconditional
    // skip made gemma's roundtrip FAIL).
    if (strcmp(tensor->name, "__missing__") == 0 && gguf_find_tensor(gguf_ctx, tensor->name) >= 0) {
        return;
    }
    // the same tensor reached through a second field (an alias) is one GGUF tensor
    for (const struct ggml_tensor * t : added_tensors) {
        if (t == tensor) {
            return;
        }
    }
    if (gguf_find_tensor(gguf_ctx, tensor->name) >= 0) {
        const std::string tensor_name = tensor->name;
        if (!(tensor_name == "rope_freqs.weight" || tensor_name == "rope_factors_long.weight" || tensor_name == "rope_factors_short.weight")) {
            LLAMA_LOG_ERROR("%s: tensor '%s' was added to the saver twice (two model fields name it)\n", __func__, tensor->name);
        }
        GGML_ASSERT(
            tensor_name == "rope_freqs.weight" || tensor_name == "rope_factors_long.weight" ||
            tensor_name == "rope_factors_short.weight"); // FIXME
        return;
    }
    gguf_add_tensor(gguf_ctx, tensor);
    added_tensors.push_back(tensor);
}

void llama_model_saver::add_kv_from_model() {
    const llama_hparams & hparams = model->hparams;
    const llama_vocab   & vocab   = model->vocab;

    const int32_t n_vocab = vocab.n_tokens();
    std::vector<std::string> tokens(n_vocab);
    std::vector<float>       scores(n_vocab);
    std::vector<int32_t>     token_types(n_vocab);

    if (vocab.get_type() != LLAMA_VOCAB_TYPE_NONE) {
        for (int32_t id = 0; id < n_vocab; ++id) {
            const llama_vocab::token_data & token_data = vocab.get_token_data(id);

            tokens[id] = token_data.text;
            scores[id] = token_data.score;

            // FIXME should this be treated as flags?
            switch(token_data.attr) {
                case LLAMA_TOKEN_ATTR_UNKNOWN:      token_types[id] = LLAMA_TOKEN_TYPE_UNKNOWN;      break;
                case LLAMA_TOKEN_ATTR_UNUSED:       token_types[id] = LLAMA_TOKEN_TYPE_UNUSED;       break;
                case LLAMA_TOKEN_ATTR_NORMAL:       token_types[id] = LLAMA_TOKEN_TYPE_NORMAL;       break;
                case LLAMA_TOKEN_ATTR_CONTROL:      token_types[id] = LLAMA_TOKEN_TYPE_CONTROL;      break;
                case LLAMA_TOKEN_ATTR_USER_DEFINED: token_types[id] = LLAMA_TOKEN_TYPE_USER_DEFINED; break;
                case LLAMA_TOKEN_ATTR_BYTE:         token_types[id] = LLAMA_TOKEN_TYPE_BYTE;         break;
                // case LLAMA_TOKEN_ATTR_NORMALIZED:   ???
                // case LLAMA_TOKEN_ATTR_LSTRIP:       ???
                // case LLAMA_TOKEN_ATTR_RSTRIP:       ???
                case LLAMA_TOKEN_ATTR_UNDEFINED:
                default:                            token_types[id] = LLAMA_TOKEN_TYPE_UNDEFINED;    break;
            }
        }
    }

    // add_kv(LLM_KV_GENERAL_TYPE,                      ???);
    add_kv(LLM_KV_GENERAL_ARCHITECTURE,              std::string(llama_model_arch_name(model->arch)));
    // add_kv(LLM_KV_GENERAL_QUANTIZATION_VERSION,      ???);
    // add_kv(LLM_KV_GENERAL_ALIGNMENT,                 ???);
    // add_kv(LLM_KV_GENERAL_FILE_TYPE,                 ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_SEQUENCE,         ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_TOP_K,            ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_TOP_P,            ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_MIN_P,            ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_XTC_PROBABILITY,  ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_XTC_THRESHOLD,    ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_TEMP,             ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_PENALTY_LAST_N,   ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_PENALTY_REPEAT,   ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_MIROSTAT,         ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_MIROSTAT_TAU,     ???);
    // add_kv(LLM_KV_GENERAL_SAMPLING_MIROSTAT_ETA,     ???);
    add_kv(LLM_KV_GENERAL_NAME,                      model->name);

    if (!model->prec_policy.prec_src1.empty()) {
        std::vector<std::string> tensor_names;
        std::vector<int8_t> values;
        tensor_names.reserve(model->prec_policy.prec_src1.size());
        values.reserve(model->prec_policy.prec_src1.size());
        for (const auto & [w, prec] : model->prec_policy.prec_src1) {
            tensor_names.push_back(ggml_get_name(w));
            values.push_back(prec == GGML_PREC_Q8 ? 0 : 1);
        }
        add_kv(LLM_KV_GENERAL_TENSOR_EXTRA_NAME, tensor_names);
        gguf_set_arr_data(gguf_ctx, llm_kv(LLM_KV_GENERAL_TENSOR_EXTRA_PREC_A4).c_str(), GGUF_TYPE_BOOL, values.data(), values.size());
    }
    // add_kv(LLM_KV_GENERAL_AUTHOR,                    ???);
    // add_kv(LLM_KV_GENERAL_VERSION,                   ???);
    // add_kv(LLM_KV_GENERAL_URL,                       ???);
    // add_kv(LLM_KV_GENERAL_DESCRIPTION,               ???);
    // add_kv(LLM_KV_GENERAL_LICENSE,                   ???);
    // add_kv(LLM_KV_GENERAL_SOURCE_URL,                ???);
    // add_kv(LLM_KV_GENERAL_SOURCE_HF_REPO,            ???);

    add_kv(LLM_KV_VOCAB_SIZE,                        vocab.n_tokens());
    add_kv(LLM_KV_CONTEXT_LENGTH,                    hparams.n_ctx_train);
    add_kv(LLM_KV_EMBEDDING_LENGTH,                  hparams.n_embd);
    if (hparams.n_embd_out > 0) {
        add_kv(LLM_KV_EMBEDDING_LENGTH_OUT,          hparams.n_embd_out);
    }
    add_kv(LLM_KV_BLOCK_COUNT,                       hparams.n_layer);
    add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
    add_kv(LLM_KV_FEED_FORWARD_LENGTH,               hparams.n_ff_arr, true);
    add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    {
        // upstream writes the per-layer array for every arch (add_kv per_layer collapses equal values to a scalar). This fork's loaders
        // fill n_ff_exp_arr only for some archs (hy_v4, whose scalar above is 0, and the upstream model classes), so the array
        // replaces the scalar only when it holds a value; an all-zero array would turn a saved model's n_ff_exp into 0
        bool any_ff_exp = false;
        for (uint32_t il = 0; il < hparams.n_layer; ++il) {
            any_ff_exp = any_ff_exp || hparams.n_ff_exp_arr[il] != 0;
        }
        if (any_ff_exp) {
            std::vector<uint32_t> n_ff_exp_per_layer(hparams.n_ff_exp_arr.begin(), hparams.n_ff_exp_arr.begin() + hparams.n_layer);
            add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff_exp_per_layer, true);
        }
    }
    add_kv(LLM_KV_EXPERT_LATENT_LENGTH,              hparams.n_expert_latent);
    add_kv(LLM_KV_MOE_LATENT_SIZE,                   hparams.moe_latent_size);
    add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    add_kv(LLM_KV_EXPERT_CHUNK_FEED_FORWARD_LENGTH,  hparams.n_ff_chexp);
    add_kv(LLM_KV_SWIGLU_CLAMP_EXP, std::vector<float>(
            hparams.swiglu_clamp_exp.begin(), hparams.swiglu_clamp_exp.begin() + hparams.n_layer));
    add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>(
            hparams.swiglu_clamp_shexp.begin(), hparams.swiglu_clamp_shexp.begin() + hparams.n_layer));
    add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,             hparams.use_par_res);
    // add_kv(LLM_KV_TENSOR_DATA_LAYOUT,                ???);
    add_kv(LLM_KV_EXPERT_COUNT,                      hparams.n_expert);
    add_kv(LLM_KV_EXPERT_USED_COUNT,                 hparams.n_expert_used_at());
    add_kv(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    add_kv(LLM_KV_EXPERT_GROUP_COUNT,                hparams.n_expert_groups);
    add_kv(LLM_KV_EXPERT_GROUP_USED_COUNT,           hparams.n_group_used);
    add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
    add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm);
    add_kv(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    add_kv(LLM_KV_EXPERT_GROUP_SCALE,                hparams.expert_group_scale);
    add_kv(LLM_KV_EXPERTS_PER_GROUP,                 hparams.n_group_experts);
    add_kv(LLM_KV_MOE_EVERY_N_LAYERS,                hparams.moe_every_n_layers);
    add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.n_layer_nextn);
    add_kv(LLM_KV_NUM_DEEPSTACK_LAYERS,              hparams.n_deepstack_layers);
    add_kv(LLM_KV_DEEPSTACK_MAPPING,                 hparams.deepstack_mapping_arr);
    add_kv(LLM_KV_POOLING_TYPE,                      uint32_t(hparams.pooling_type));
    add_kv(LLM_KV_LOGIT_SCALE,                       hparams.f_logit_scale);
    // the T5 loader reads this key as uint32 (llama-hparams.cpp, T5 case; models/t5.cpp:8) and get_key rejects any other GGUF type
    // (llama-model-loader.cpp:219-226), but dec_start_token_id is a llama_token (int32, llama-hparams.h:476), so the generic add_kv wrote
    // an INT32 and a saved t5 stopped at "wrong type i32 but expected type u32". Written as the converter writes it (u32), and only when
    // set: absent, every loader keeps the default -1, which is what an unset field holds.
    if (hparams.dec_start_token_id >= 0) {
        add_kv(LLM_KV_DECODER_START_TOKEN_ID,        uint32_t(hparams.dec_start_token_id));
    }
    add_kv(LLM_KV_DECODER_BLOCK_COUNT,               hparams.dec_n_layer);
    add_kv(LLM_KV_ATTN_LOGIT_SOFTCAPPING,            hparams.f_attn_logit_softcapping);
    add_kv(LLM_KV_ROUTER_LOGIT_SOFTCAPPING,          hparams.f_router_logit_softcapping);
    add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING,           hparams.f_final_logit_softcapping);
    add_kv(LLM_KV_SWIN_NORM,                         hparams.swin_norm);
    add_kv(LLM_KV_RESCALE_EVERY_N_LAYERS,            hparams.rescale_every_n_layers);
    add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,                hparams.time_mix_extra_dim);
    add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,              hparams.time_decay_extra_dim);
    add_kv(LLM_KV_RESIDUAL_SCALE,                    hparams.f_residual_scale);
    add_kv(LLM_KV_EMBEDDING_SCALE,                   hparams.f_embedding_scale);
    add_kv(LLM_KV_HRM_LAYERS_PER_STACK,              hparams.n_hrm_layers_per_stack);
    add_kv(LLM_KV_HRM_H_CYCLES,                      hparams.n_hrm_h_cycles);
    add_kv(LLM_KV_HRM_L_CYCLES,                      hparams.n_hrm_l_cycles);
    add_kv(LLM_KV_HRM_PREFIX_LM,                     hparams.hrm_prefix_lm);
    add_kv(LLM_KV_TOKEN_SHIFT_COUNT,                 hparams.token_shift_count);
    add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,         hparams.n_moe_layer_step);
    // The qwen3next / qwen35 / qwen4exp loaders (llama-hparams.cpp) derive their recurrent layers ONLY from
    // LLM_KV_FULL_ATTENTION_INTERVAL and never read LLM_KV_ATTENTION_RECURRENT_LAYERS, so a saved file that omits the
    // interval reloads with the default of 4 and a different layer layout (test-llama-archs -a qwen35, 2 layers:
    // layer 1 came back recurrent and its ssm_conv1d tensor was asked for). Write the interval when the layout is periodic.
    if (model != nullptr && (model->arch == LLM_ARCH_QWEN3NEXT || model->arch == LLM_ARCH_QWEN35 ||
                             model->arch == LLM_ARCH_QWEN35MOE || model->arch == LLM_ARCH_QWEN4EXP)) {
        const uint32_t n_main = hparams.n_layer - hparams.nextn_predict_layers;
        for (uint32_t k = 2; k <= n_main; ++k) {
            bool periodic = true;
            for (uint32_t il = 0; il < n_main && periodic; ++il) {
                periodic = hparams.is_recr_impl[il] == ((il + 1) % k != 0);
            }
            if (periodic) {
                add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, k);
                break;
            }
        }
    }

    add_kv(LLM_KV_ATTENTION_HEAD_COUNT,              hparams.n_head_arr, true);
    add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,           hparams.n_head_kv_arr, true);
    add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,          hparams.f_max_alibi_bias);
    add_kv(LLM_KV_ATTENTION_CLAMP_KQV,               hparams.f_clamp_kqv);
    if (model->arch == LLM_ARCH_GLM5NEXT || model->arch == LLM_ARCH_GLM5NEXT_DASHED) {
        // SOURCE-READ: upstream/unsloth-llama.cpp/conversion/glm5next.py (no code copied; the key semantics only): the converter writes
        // key_length = kv_lora_rank + qk_rope_head_dim and value_length = kv_lora_rank, the ABSORBED latent width of the K/V cache. The GLM5NEXT
        // loader then overwrites n_embd_head_k_full / _v_full with the MLA lengths, so the generic fields no longer hold what the file carried;
        // writing them back would give the Unsloth engine a cache 192 wide against a 512-wide latent (ggml_set_rows assert, measured 2026-10-11).
        add_kv(LLM_KV_ATTENTION_KEY_LENGTH,          hparams.n_lora_kv + hparams.n_rot);
        add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,        hparams.n_lora_kv);
    } else {
        add_kv(LLM_KV_ATTENTION_KEY_LENGTH,          hparams.n_embd_head_k_full);
        add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,        hparams.n_embd_head_v_full);
    }
    add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,           hparams.f_norm_eps);
    add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    add_kv(LLM_KV_POST_NORM_EPS,                     hparams.f_post_norm_eps);
    add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,           hparams.f_norm_group_eps);
    add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,        hparams.n_norm_groups);
    add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      hparams.n_value_expert);
    add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, hparams.n_value_expert_used);
    add_kv(LLM_KV_ATTENTION_CAUSAL,                  hparams.causal_attn);
    add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,             hparams.n_lora_q);
    add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,            hparams.n_lora_kv);
    add_kv(LLM_KV_ATTENTION_DECAY_LORA_RANK,         hparams.n_lora_decay);
    add_kv(LLM_KV_ATTENTION_ICLR_LORA_RANK,          hparams.n_lora_iclr);
    add_kv(LLM_KV_ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK, hparams.n_lora_value_res_mix);
    add_kv(LLM_KV_ATTENTION_GATE_LORA_RANK,          hparams.n_lora_gate);
    add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT,  hparams.n_rel_attn_bkts);
    add_kv(LLM_KV_ATTENTION_ROPE_PATTERN,            hparams.rope_pattern, true);
    add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,          hparams.n_swa);
    // llama.cpp #29042 (542348a35c9e, MIT): the pattern is written as one flag per layer for EVERY SWA model -- over hparams.n_layer,
    // which in this fork is the total incl. nextn layers (upstream n_layer_all) -- and never collapsed to a scalar (per_layer=false),
    // because a scalar is read back as a period. Every loader that used to read only a period now reads either form
    // (llm_read_swa_pattern / llm_load_swa_pattern, llama-hparams.cpp); since saverrest-1019 that includes the legacy GEMMA3/COHERE2/LLAMA4
    // cases, whose graphs now read hparams.is_swa(il) instead of a hard-coded period. is_swa_any() keeps the gemma4/diffusion-gemma/spark2_5
    // case that wrote it before this change (DiffusionGemma: required array, test-llama-archs -a diffusion-gemma, 2026-10-04).
    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE || hparams.is_swa_any()) {
        std::vector<uint32_t> pattern(hparams.n_layer);
        for (uint32_t il = 0; il < hparams.n_layer; il++) {
            pattern[il] = hparams.is_swa_impl[il] ? 1 : 0;
        }
        add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern, false);
    }
    if (model->arch == LLM_ARCH_GEMMA4_MTP) {
        // the MTP loader requires the backbone width and reads the rest as optional; none were written, so a saved model could not be loaded back
        add_kv(LLM_KV_MTP_BACKBONE_EMBEDDING_LENGTH, hparams.mtp_backbone_n_embd);
        add_kv(LLM_KV_MTP_CENTROID_COUNT,            hparams.mtp_num_centroids);
        add_kv(LLM_KV_MTP_CENTROID_TOP_K,            hparams.mtp_centroid_top_k);
        add_kv(LLM_KV_MTP_USE_ORDERED_EMBEDDINGS,    hparams.mtp_use_ordered_embeddings);
    }
    if (model->arch == LLM_ARCH_GEMMA4) {
        // REQUIRED by the gemma4 loader (get_key without a not-required flag); the saver did not write it, so a saved gemma4 could not be loaded back
        add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
    }
    if (model->arch == LLM_ARCH_DIFFUSION_GEMMA) {
        // read by llama-hparams.cpp as a raw key (no arch prefix) and REQUIRED; there is no llm_kv for it
        gguf_set_val_u32(gguf_ctx, "diffusion.canvas_length", hparams.diffusion_canvas_length);
    }
    add_kv(LLM_KV_ATTENTION_SCALE,                   hparams.f_attention_scale);
    add_kv(LLM_KV_ATTENTION_OUTPUT_SCALE,            hparams.f_attn_out_scale);
    add_kv(LLM_KV_ATTENTION_VALUE_SCALE,             hparams.f_attn_v_scale);
    add_kv(LLM_KV_ATTENTION_TEMPERATURE_LENGTH,      hparams.attn_temp_length);
    add_kv(LLM_KV_ATTENTION_TEMPERATURE_SCALE,       hparams.f_attn_temp_scale);
    // The bailingmoe3 and kimi_k3 loaders (llama-hparams.cpp, ml.get_key(LLM_KV_ATTENTION_*_LENGTH_MLA, hparams.n_embd_head_*_full)) read the
    // MLA keys INTO the full head lengths and never set n_embd_head_*_mla_impl, so the saver wrote 0 and the reload got n_embd_head_k_full = 0:
    // check_tensor_dims "blk.0.attn_q_b.weight ... expected 512, 0, got 512, 192" (measured 2026-10-07, bailingmoe3 round trip).
    // glm5next (llama-hparams.cpp, GLM5NEXT case) does the same: it reads KEY_LENGTH_MLA / VALUE_LENGTH_MLA into n_embd_head_k_full / n_embd_head_v_full
    // and kda.head_dim into ssm_d_state, so its saved file needs the full-field values under those keys too (reload: attn_q_b expected 512 x 0).
    const bool hparams_loader_reads_into_full_fields = model->arch == LLM_ARCH_BAILINGMOE3 || model->arch == LLM_ARCH_KIMI_K3 ||
                                                       model->arch == LLM_ARCH_GLM5NEXT || model->arch == LLM_ARCH_GLM5NEXT_DASHED;
    add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,          hparams_loader_reads_into_full_fields ? hparams.n_embd_head_k_full : hparams.n_embd_head_k_mla_impl);
    add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,        hparams_loader_reads_into_full_fields ? hparams.n_embd_head_v_full : hparams.n_embd_head_v_mla_impl);
    add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,          hparams.n_embd_head_k_swa);
    add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,        hparams.n_embd_head_v_swa);
    // #29042: MLA on the SWA layers (dots3note reads all three as REQUIRED, models/dots3note.cpp:33-35)
    add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,      hparams.n_embd_head_k_mla_swa);
    add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA,    hparams.n_embd_head_v_mla_swa);
    add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,        hparams.n_lora_kv_swa);
    add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,      hparams.indexer_n_head);
    add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,      hparams.indexer_head_size);
    add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,           hparams.indexer_top_k);
    add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,      hparams.indexer_block_size);
    add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,           hparams.indexer_kpool);
    add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, hparams.indexer_kpool_select_tail);
    if (model->arch == LLM_ARCH_GLM5NEXT || model->arch == LLM_ARCH_GLM5NEXT_DASHED) {
        // the glm5next loader (llama-hparams.cpp, GLM5NEXT case) reads the pool width from attention.indexer.kpool and not from
        // attention.indexer.block_size, so the line above leaves a saved file with indexer_block_size = 0 (no pooling) on reload
        add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,       hparams.indexer_block_size);
    }
    add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,    hparams.indexer_local_blocks);
    add_kv(LLM_KV_ATTENTION_INDEXER_TYPES,           hparams.is_indexer_full_impl, true);
    add_kv(LLM_KV_ATTENTION_RECURRENT_LAYERS,        hparams.is_recr_impl, true);
    add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,      hparams.dsv4_o_group_count);
    add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,        hparams.dsv4_o_lora_rank);
    add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, hparams.dsv4_compress_rope_base);
    if (llm_arch_is_dsv4(model->arch) || hparams.dsv4_hc_mult > 0) {
        // the loader requires one compress ratio per layer, including nextn layers
        const std::vector<uint32_t> compress_ratios(
                hparams.dsv4_compress_ratios.begin(), hparams.dsv4_compress_ratios.begin() + hparams.n_layer);
        add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,         compress_ratios);
    } else {
        add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,         hparams.dsv4_compress_ratios, true);
    }
    add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE,           hparams.hc_magnitude);
    if (model->arch == LLM_ARCH_QWEN4EXP) {
        // REQUIRED by the qwen4exp loader (llama-hparams.cpp, QWEN4EXP case: ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank)),
        // never written before, so a saved qwen4exp stopped at "key not found: qwen4exp.hyper_connection.low_rank". hc_low_rank sizes every
        // hc_*_down / hc_*_up tensor (llama-load-tensors.cpp, create_qwen4exp_tensors), so it cannot be left to a default either.
        add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK,        hparams.hc_low_rank);
        // the qwen4exp loader reads nextn_predict_layers into hparams.nextn_predict_layers and never syncs n_layer_nextn (llama-hparams.h:197-225,
        // "two fields, one GGUF key"), so the generic line above wrote 0 and a saved model with a NextN tail reloaded without it
        add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,             hparams.nextn_predict_layers);
        if (hparams.ple_n_heads > 0) {
            // the n-gram (PLE) module: the loader reads it only when qwen4exp.ple.layers is present and non-empty, and then REQUIRES the
            // geometry keys and the three uint64 hash arrays (llama-hparams.cpp, QWEN4EXP case, "ple.layers is absent ...")
            std::vector<uint32_t> ple_layers;
            for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                if (hparams.ple_layer_arr[il]) {
                    ple_layers.push_back(il);
                }
            }
            add_kv(LLM_KV_PLE_LAYERS,                    ple_layers);
            add_kv(LLM_KV_PLE_NGRAM_SIZE,                hparams.ple_ngram_size);
            add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,           hparams.ple_heads_per_ngram);
            add_kv(LLM_KV_PLE_CONV_KERNEL,               hparams.ple_conv_kernel);
            add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,    hparams.ple_head_dim);
            add_kv(LLM_KV_PLE_EOS_TOKEN_ID,              hparams.ple_eos_token_id);
            add_kv(LLM_KV_PLE_IMAGE_TOKEN_ID,            hparams.ple_image_token_id);
            // the template add_kv has no uint64 branch; the loader reads these with get_arr<uint64_t>, which accepts GGUF_TYPE_UINT64
            gguf_set_arr_data(gguf_ctx, llm_kv(LLM_KV_PLE_LAYER_MULTIPLIERS).c_str(), GGUF_TYPE_UINT64,
                    hparams.ple_layer_multipliers.data(), hparams.ple_ngram_size);
            gguf_set_arr_data(gguf_ctx, llm_kv(LLM_KV_PLE_HEAD_OFFSETS).c_str(),      GGUF_TYPE_UINT64,
                    hparams.ple_head_offsets.data(),      hparams.ple_n_heads);
            gguf_set_arr_data(gguf_ctx, llm_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES).c_str(),  GGUF_TYPE_UINT64,
                    hparams.ple_head_vocab_sizes.data(),  hparams.ple_n_heads);
        }
    }
    if (model->arch == LLM_ARCH_OPENPANGU) {
        // REQUIRED by the openpangu loader (llama-hparams.cpp, OPENPANGU case, "mHC / Hyper-Connections + learned param sink"), never written
        // before, so a saved openpangu stopped at "key not found: openpangu.mhc_num_stream"
        add_kv(LLM_KV_OPENPANGU_MHC_NUM_STREAM,         hparams.mhc_num_stream);
        add_kv(LLM_KV_OPENPANGU_MHC_RECUR_NORM,         hparams.mhc_recur_norm);
        add_kv(LLM_KV_OPENPANGU_PARAM_SINK_NUMBER,      hparams.param_sink_number);
        if (hparams.is_swa_any()) {
            // the DSA/SWA schedule is read ONLY from openpangu.swa_layers + openpangu.sliding_window_list (raw keys, no llm_kv); without them
            // every layer reloads as DSA/dense. Each listed layer carries its window: n_swa for a base layer, n_swa_mtp for a NextN layer.
            const uint32_t n_base = hparams.n_layer > hparams.nextn_predict_layers ? hparams.n_layer - hparams.nextn_predict_layers : hparams.n_layer;
            std::vector<uint32_t> swa_ids, swa_windows;
            for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                if (hparams.is_swa_impl[il]) {
                    swa_ids.push_back(il);
                    swa_windows.push_back(il < n_base ? hparams.n_swa : hparams.n_swa_mtp);
                }
            }
            gguf_set_arr_data(gguf_ctx, "openpangu.swa_layers",          GGUF_TYPE_UINT32, swa_ids.data(),     swa_ids.size());
            gguf_set_arr_data(gguf_ctx, "openpangu.sliding_window_list", GGUF_TYPE_UINT32, swa_windows.data(), swa_windows.size());
        }
    }
    add_kv(LLM_KV_HASH_LAYER_COUNT,                     hparams.dsv4_hash_layer_count);
    add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK,             hparams.hc_low_rank);

    if (model->arch == LLM_ARCH_DEEPSEEK41) {
        const auto * m41 = dynamic_cast<const llama_model_deepseek41 *>(model);
        GGML_ASSERT(m41 != nullptr);
        std::vector<uint32_t> kv_src;
        std::vector<uint32_t> idx_src;
        for (uint32_t il = 0; il < hparams.n_layer_eff(); ++il) {
            if (hparams.dsv41_kv_src[il] == (int32_t) il) {
                kv_src.push_back(il);
            }
            if (hparams.dsv41_topk_src[il] == (int32_t) il) {
                idx_src.push_back(il);
            }
        }
        add_kv(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,      kv_src);
        add_kv(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS, idx_src);
        if (hparams.dsv41_cand_src >= 0) {
            add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, hparams.dsv41_cand_src);
            add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, hparams.dsv41_cand_topk_blocks);
            add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   hparams.dsv41_cand_block_size);
        }
        if (hparams.ple_n_heads > 0) {
            add_kv(LLM_KV_ENGRAM_PAD_TOKEN_ID, m41->engram_pad_id);
            add_kv(LLM_KV_ENGRAM_TOKEN_MAP,    m41->engram_token_map);
        }
    }

    // the PLE group only means anything whole: write all of it or none
    if (hparams.ple_n_heads > 0) {
        std::vector<uint32_t> ple_layers;
        for (uint32_t il = 0; il < hparams.n_layer; ++il) {
            if (hparams.is_ple_impl[il]) {
                ple_layers.push_back(il);
            }
        }
        add_kv(LLM_KV_PLE_LAYERS,            ple_layers);
        add_kv(LLM_KV_PLE_NGRAM_SIZE,        hparams.ple_ngram_size);
        add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,   hparams.ple_heads_per_ngram);
        add_kv(LLM_KV_PLE_CONV_KERNEL,       hparams.ple_conv_kernel);
        add_kv(LLM_KV_PLE_EOS_TOKEN_ID,      hparams.ple_eos_token_id);
        add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.ple_head_dim);
        add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS, std::vector<uint64_t>(
                hparams.ple_layer_multipliers.begin(),
                hparams.ple_layer_multipliers.begin() + hparams.ple_n_modules*hparams.ple_ngram_size));
        add_kv(LLM_KV_PLE_HEAD_OFFSETS,      std::vector<uint64_t>(
                hparams.ple_head_offsets.begin(),
                hparams.ple_head_offsets.begin() + hparams.ple_n_modules*hparams.ple_n_heads));
        add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,  std::vector<uint64_t>(
                hparams.ple_head_vocab_sizes.begin(),
                hparams.ple_head_vocab_sizes.begin() + hparams.ple_n_modules*hparams.ple_n_heads));
    }

    const float rope_scaling_factor = hparams.rope_freq_scale_train == 1.0f ? 0.0f : 1.0f/hparams.rope_freq_scale_train;

    add_kv(LLM_KV_ROPE_DIMENSION_COUNT,              hparams.n_rot);
    add_kv(LLM_KV_ROPE_DIMENSION_COUNT_SWA,          hparams.n_rot_swa);
    add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,           hparams.rope_sections);
    add_kv(LLM_KV_ROPE_SECTION_ORDER,                llama_rope_section_order_name(hparams.rope_section_order));
    add_kv(LLM_KV_ROPE_FREQ_BASE,                    hparams.rope_freq_base_train);
    add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,                hparams.rope_freq_base_train_swa);
    // add_kv(LLM_KV_ROPE_SCALE_LINEAR,                 rope_scaling_factor); // old name
    add_kv(LLM_KV_ROPE_SCALING_TYPE,                 llama_rope_scaling_type_name(hparams.rope_scaling_type_train));
    add_kv(LLM_KV_ROPE_SCALING_FACTOR,               rope_scaling_factor);
    add_kv(LLM_KV_ROPE_SCALING_ATTN_FACTOR,          hparams.rope_attn_factor);
    add_kv(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,         hparams.n_ctx_orig_yarn);
    add_kv(LLM_KV_ROPE_SCALING_FINETUNED,            hparams.rope_finetuned);
    add_kv(LLM_KV_ROPE_SCALING_YARN_LOG_MUL,         hparams.rope_yarn_log_mul);
    add_kv(LLM_KV_ROPE_SCALING_YARN_MSCALE,          hparams.rope_yarn_mscale);
    add_kv(LLM_KV_ROPE_SCALING_YARN_TRUNCATE,        hparams.rope_yarn_truncate);
    add_kv(LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR,      hparams.yarn_ext_factor);
    add_kv(LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR,     hparams.yarn_attn_factor);
    add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_FAST,       hparams.yarn_beta_fast);
    add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,       hparams.yarn_beta_slow);

    // TODO: implement split file support
    // add_kv(LLM_KV_SPLIT_NO,                          ???);
    // add_kv(LLM_KV_SPLIT_COUNT,                       ???);
    // add_kv(LLM_KV_SPLIT_TENSORS_COUNT,               ???);

    add_kv(LLM_KV_SSM_INNER_SIZE,                    hparams.ssm_d_inner);
    add_kv(LLM_KV_SSM_CONV_KERNEL,                   hparams.ssm_d_conv);
    add_kv(LLM_KV_SSM_STATE_SIZE,                    hparams.ssm_d_state);
    add_kv(LLM_KV_SSM_TIME_STEP_RANK,                hparams.ssm_dt_rank);
    add_kv(LLM_KV_SSM_GROUP_COUNT,                   hparams.ssm_n_group);
    add_kv(LLM_KV_SSM_DT_B_C_RMS,                    hparams.ssm_dt_b_c_rms);

    // the same two archs' hparams loaders (llama-hparams.cpp:2037, :2089) read kda.head_dim into ssm_d_state, so n_embd_head_kda stays 0 for them
    // and the saver wrote kda.head_dim = 0 (measured 2026-10-07: reloaded file expected attn_output 0 x 128).
    add_kv(LLM_KV_KDA_HEAD_DIM,                      hparams_loader_reads_into_full_fields ? hparams.ssm_d_state : hparams.n_embd_head_kda);
    add_kv(LLM_KV_KDA_SAFE_GATE,                     hparams.kda_safe_gate);
    add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,              hparams.kda_gate_lower_bound);

    add_kv(LLM_KV_WKV_HEAD_SIZE,                     hparams.wkv_head_size);

    add_kv(LLM_KV_TOKENIZER_MODEL,                   vocab.get_tokenizer_model());
    add_kv(LLM_KV_TOKENIZER_PRE,                     vocab.get_tokenizer_pre());
    add_kv(LLM_KV_TOKENIZER_LIST,                    tokens);
    add_kv(LLM_KV_TOKENIZER_TOKEN_TYPE,              token_types);
    add_kv(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,        vocab.n_token_types());
    add_kv(LLM_KV_TOKENIZER_SCORES,                  scores);
    add_kv(LLM_KV_TOKENIZER_MERGES,                  vocab.get_bpe_merges());
    // FIXME llama_token is type i32 but when reading in a GGUF file u32 is expected, not an issue for writing though
    if (vocab.token_bos()  != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_BOS_ID, uint32_t(vocab.token_bos()));  }
    if (vocab.token_eos()  != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_EOS_ID, uint32_t(vocab.token_eos()));  }
    if (vocab.token_eot()  != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_EOT_ID, uint32_t(vocab.token_eot()));  }
    if (vocab.token_eom()  != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_EOM_ID, uint32_t(vocab.token_eom()));  }
    if (vocab.token_unk()  != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_UNK_ID, uint32_t(vocab.token_unk()));  }
    if (vocab.token_sep()  != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_SEP_ID, uint32_t(vocab.token_sep()));  }
    if (vocab.token_pad()  != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_PAD_ID, uint32_t(vocab.token_pad()));  }
    // add_kv(LLM_KV_TOKENIZER_CLS_ID,                  uint32_t(vocab.token_bos())); // deprecated
    // add_kv(LLM_KV_TOKENIZER_MASK_ID,                 ???);
    add_kv(LLM_KV_TOKENIZER_ADD_BOS,                 vocab.get_add_bos());
    add_kv(LLM_KV_TOKENIZER_ADD_EOS,                 vocab.get_add_eos());
    add_kv(LLM_KV_TOKENIZER_ADD_SEP,                 vocab.get_add_sep());
    add_kv(LLM_KV_TOKENIZER_ADD_PREFIX,              vocab.get_add_space_prefix());
    add_kv(LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,         vocab.get_remove_extra_whitespaces());
    add_kv(LLM_KV_TOKENIZER_CLEAN_SPACES,            vocab.get_clean_spaces());
    add_kv(LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP,    vocab.get_precompiled_charsmap());
    // add_kv(LLM_KV_TOKENIZER_HF_JSON,                 ???);
    // add_kv(LLM_KV_TOKENIZER_RWKV,                    ???);
    if (vocab.token_fim_pre() != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_FIM_PRE_ID, uint32_t(vocab.token_fim_pre())); }
    if (vocab.token_fim_suf() != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_FIM_SUF_ID, uint32_t(vocab.token_fim_suf())); }
    if (vocab.token_fim_mid() != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_FIM_MID_ID, uint32_t(vocab.token_fim_mid())); }
    if (vocab.token_fim_pad() != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_FIM_PAD_ID, uint32_t(vocab.token_fim_pad())); }
    if (vocab.token_fim_rep() != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_FIM_REP_ID, uint32_t(vocab.token_fim_rep())); }
    if (vocab.token_fim_sep() != LLAMA_TOKEN_NULL) { add_kv(LLM_KV_TOKENIZER_FIM_SEP_ID, uint32_t(vocab.token_fim_sep())); }

    // TODO: implement LoRA support
    // add_kv(LLM_KV_ADAPTER_TYPE,                      ???);
    // add_kv(LLM_KV_ADAPTER_LORA_ALPHA,                ???);
    // add_kv(LLM_KV_ADAPTER_LORA_TASK_NAME,            ???);
    // add_kv(LLM_KV_ADAPTER_LORA_PROMPT_PREFIX,        ???);
    // add_kv(LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS,   ???);

    add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,           hparams.posnet.n_embd);
    add_kv(LLM_KV_POSNET_BLOCK_COUNT,                hparams.posnet.n_layer);

    add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH,         hparams.convnext.n_embd);
    add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,              hparams.convnext.n_layer);

    add_kv(LLM_KV_CLASSIFIER_OUTPUT_LABELS,          model->classifier_labels);

    add_kv(LLM_KV_SHORTCONV_L_CACHE,                 hparams.n_shortconv_l_cache);

    // per layer (n_layer_eff values, a scalar when all equal): the apertus loader (models/apertus.cpp:6-9) reads these with
    // get_key_or_arr(..., n_layer_eff()), which rejects an array of any other length (llama-model-loader.cpp:1688-1694). Written whole,
    // the std::array went out as LLAMA_MAX_LAYERS = 512 floats and a saved apertus could not be loaded back. No other loader reads them.
    add_kv(LLM_KV_XIELU_ALPHA_N,                     hparams.xielu_alpha_n, true);
    add_kv(LLM_KV_XIELU_ALPHA_P,                     hparams.xielu_alpha_p, true);
    add_kv(LLM_KV_XIELU_BETA,                        hparams.xielu_beta,    true);
    add_kv(LLM_KV_XIELU_EPS,                         hparams.xielu_eps,     true);

    add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,               hparams.attn_res_block_size);
    add_kv(LLM_KV_ACTIVATION_SITU_BETA,              hparams.situ_beta);
    add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA,       hparams.situ_linear_beta);

    // deprecated
    // add_kv(LLM_KV_TOKENIZER_PREFIX_ID,               ???);
    // add_kv(LLM_KV_TOKENIZER_SUFFIX_ID,               ???);
    // add_kv(LLM_KV_TOKENIZER_MIDDLE_ID,               ???);

    add_kv(LLM_KV_DENSE_2_FEAT_IN,                   hparams.dense_2_feat_in);
    add_kv(LLM_KV_DENSE_2_FEAT_OUT,                  hparams.dense_2_feat_out);
    add_kv(LLM_KV_DENSE_3_FEAT_IN,                   hparams.dense_3_feat_in);
    add_kv(LLM_KV_DENSE_3_FEAT_OUT,                  hparams.dense_3_feat_out);
}

void llama_model_saver::add_tensors_from_model() {
    if (model->output != nullptr &&
            std::string(model->output->name) != std::string(model->tok_embd->name)) {
        add_tensor(model->tok_embd); // some models use the same tensor for tok_embd and output
    }
    add_tensor(model->type_embd);
    add_tensor(model->pos_embd);
    add_tensor(model->tok_norm);
    add_tensor(model->tok_norm_b);
    // model-level tensors of the gemma4 family, absent from this list until 2026-10-04: a saved gemma4 failed to load back with
    // "tensor 'per_layer_token_embd.weight' not found"
    add_tensor(model->tok_embd_per_layer);
    add_tensor(model->per_layer_model_proj);
    add_tensor(model->per_layer_proj_norm);
    add_tensor(model->mtp_pre_proj);
    add_tensor(model->mtp_post_proj);
    add_tensor(model->mtp_token_ordering);
    add_tensor(model->mtp_centroids);
    add_tensor(model->output_norm);
    add_tensor(model->output_norm_b);
    add_tensor(model->output);
    add_tensor(model->output_b);
    add_tensor(model->output_norm_enc);
    add_tensor(model->output_s);
    add_tensor(model->output_in_s);
    add_tensor(model->output_res_score);
    add_tensor(model->nextn_proj_pre);
    add_tensor(model->nextn_proj_post);
    add_tensor(model->cls);
    add_tensor(model->cls_b);
    add_tensor(model->cls_out);
    add_tensor(model->cls_out_b);
    add_tensor(model->cls_norm);
    add_tensor(model->hrm_z_l_init);
    add_tensor(model->hc_head_fn);
    add_tensor(model->hc_head_base);
    add_tensor(model->hc_head_scale);
    // qwen4exp's final low-rank hyper-connection mix (output_hc_norm / _down / _up, llama-load-tensors.cpp create_qwen4exp_tensors) and openpangu's
    // global mHC stream merge (mhc_merge_*, create_openpangu_tensors): model-level fields this list never named, so a saved file lacked them
    add_tensor(model->hc_head_norm);
    add_tensor(model->hc_head_down);
    add_tensor(model->hc_head_up);
    add_tensor(model->mhc_merge_phi);
    add_tensor(model->mhc_merge_alpha);
    add_tensor(model->mhc_merge_beta);
    add_tensor(model->mhc_merge_gamma);
    // gemma3n's model-level AltUp projections and per-layer token embedding (models/gemma3n.cpp:36-39, fields llama-model.h:1134-1136,
    // distinct from the gemma4 tok_embd_per_layer above): no line named them, so a saved gemma3n lacked three required tensors.
    // llama.cpp master saves per_layer_tok_embd (llama-model-saver.cpp, add_tensors_from_model; MIT) but not the two AltUp ones.
    add_tensor(model->altup_proj);
    add_tensor(model->altup_unembd_proj);
    add_tensor(model->per_layer_tok_embd);

    // looped architectures alias physical tensors across cache slots; save each
    // tensor once. a different tensor with an existing name still asserts below
    std::unordered_set<const struct ggml_tensor *> seen;

    for (const struct llama_layer & layer : model->layers) {
        // Some architectures point two llama_layer fields at ONE tensor (qwen35: ffn_norm = attn_post_norm), which
        // is a single GGUF tensor, so add each distinct tensor once instead of tripping add_tensor()'s duplicate-name assert.
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            const struct ggml_tensor * t = reinterpret_cast<const struct ggml_tensor * const *>(&layer)[i];
            // upstream: looped architectures alias one physical tensor across several llama_layer entries, so `seen` spans every layer
            // (it also covers two fields of one layer naming one tensor, qwen35 ffn_norm = attn_post_norm)
            if (t == nullptr || seen.count(t) != 0) {
                continue;
            }
            // bailingmoe3 / kimi_k3 read wk_b + wv_b from the file; the combined wkv_b is computed from them at load (computed_wkv_b), so
            // writing it adds a tensor the loader never asks for: "done_getting_tensors: wrong number of tensors; expected 41, got 40"
            // (measured 2026-10-07, blk.0.attn_kv_b.weight, Q8_0, the one extra tensor in the saved file).
            // glm5next does the same (llm_prepare_mla: "need to compute 1 wkv_b tensors" at load): the saved file then held a Q8_0 blk.0.attn_kv_b.weight
            // and the reload stopped at "wrong number of tensors; expected 61, got 60" (measured 2026-10-11, test-llama-archs -a glm5next).
            // glm-dsa is the same case (create_glm_dsa_tensors reads only wk_b + wv_b; llm_prepare_mla computes wkv_b into computed_wkv_b):
            // the saved file held 46 tensors and the reload stopped at "wrong number of tensors; expected 46, got 44" (test-llama-archs
            // -a glm-dsa, 2026-10-19, "need to compute 2 wkv_b tensors"). Any wkv_b that IS the load-computed one is skipped, whatever the arch.
            if (t != nullptr && t == layer.computed_wkv_b.get()) {
                continue;
            }
            if (t == layer.wkv_b && (model->arch == LLM_ARCH_BAILINGMOE3 || model->arch == LLM_ARCH_KIMI_K3 ||
                                     model->arch == LLM_ARCH_GLM5NEXT || model->arch == LLM_ARCH_GLM5NEXT_DASHED)) {
                continue;
            }
            // openpangu's attn_param_sink_blk / attn_param_sink_lat_t are DERIVED at load from param_sink_kv, param_sink_k_pe and attn_kv_a_norm
            // (llama.cpp, llm_prepare_openpangu_param_sinks) and owned by these two members; create_openpangu_tensors never asks for them, so
            // writing them would leave the reload with two unrequested tensors per layer
            if (t != nullptr && (t == layer.computed_param_sink_blk.get() || t == layer.computed_param_sink_lat_t.get())) {
                continue;
            }
            seen.insert(t);
            add_tensor(t);
        }
    }
}

// llm_scale_gate_inp_s (llama.cpp) multiplies every layer's ffn_gate_inp_s by 1/sqrtf(n_embd) IN PLACE when a gemma4 / diffusion-gemma model loads, so the
// memory holds the folded value while the file format stores the unfolded one. Writing the folded value made a saved model scale it a second time on reload:
// test-llama-archs -a diffusion-gemma, reloaded model vs device model, first differing node ffn_moe_router_in-0 at relative error 0.91 (2026-10-04).
// While a file is written, substitute for each element the float whose load-time scaling gives back exactly the in-memory value (`y * scale == z`, searched
// among the neighbours of z / scale because that product does not round-trip bit for bit), then restore the originals.
namespace {
struct gate_inp_s_unfold {
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> original;

    explicit gate_inp_s_unfold(const llama_model * model) {
        if (model == nullptr || (model->arch != LLM_ARCH_GEMMA4 && model->arch != LLM_ARCH_DIFFUSION_GEMMA)) {
            return;
        }
        const float scale = 1.0f/sqrtf((float) model->hparams.n_embd); // the loader's expression
        for (const llama_layer & l : model->layers) {
            ggml_tensor * t = l.ffn_gate_inp_s;
            if (t == nullptr || t->type != GGML_TYPE_F32 || t->extra != nullptr || t->buffer == nullptr) {
                continue;
            }
            std::vector<float> z(ggml_nelements(t));
            ggml_backend_tensor_get(t, z.data(), 0, ggml_nbytes(t));
            std::vector<float> y(z.size());
            for (size_t i = 0; i < z.size(); i++) {
                float c = z[i] / scale;
                float best = c;
                for (int k = 0; k < 4 && best * scale != z[i]; k++) { // walk outwards: c, c+1ulp, c-1ulp, c+2ulp, c-2ulp
                    const float up = nextafterf(c, INFINITY), dn = nextafterf(c, -INFINITY);
                    const float cand[4] = { up, dn, nextafterf(up, INFINITY), nextafterf(dn, -INFINITY) };
                    best = cand[k];
                }
                y[i] = (best * scale == z[i]) ? best : c;
            }
            original.emplace_back(t, std::move(z));
            ggml_backend_tensor_set(t, y.data(), 0, ggml_nbytes(t));
        }
    }

    ~gate_inp_s_unfold() {
        for (auto & p : original) {
            ggml_backend_tensor_set(p.first, p.second.data(), 0, ggml_nbytes(p.first));
        }
    }
};
} // namespace

void llama_model_saver::save(const std::string & path_model) {
    gate_inp_s_unfold unfold(model);
    gguf_write_to_file(gguf_ctx, path_model.c_str(), false);
}

void llama_model_saver::save(FILE * file) {
    gate_inp_s_unfold unfold(model);
    gguf_write_to_file_ptr(gguf_ctx, file, false);
}
