#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <map>

static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,           "llama"        },
    { LLM_ARCH_K2_HORIZON,      "k2-horizon"   },
    { LLM_ARCH_LLAMA4,          "llama4"       },
    { LLM_ARCH_DECI,            "deci"         },
    { LLM_ARCH_FALCON,          "falcon"       },
    { LLM_ARCH_GROK,            "grok"         },
    { LLM_ARCH_GPT2,            "gpt2"         },
    { LLM_ARCH_GPTJ,            "gptj"         },
    { LLM_ARCH_GPTNEOX,         "gptneox"      },
    { LLM_ARCH_MPT,             "mpt"          },
    { LLM_ARCH_BAICHUAN,        "baichuan"     },
    { LLM_ARCH_STARCODER,       "starcoder"    },
    { LLM_ARCH_REFACT,          "refact"       },
    { LLM_ARCH_BERT,            "bert"         },
    { LLM_ARCH_NOMIC_BERT,      "nomic-bert"   },
    { LLM_ARCH_JINA_BERT_V2,    "jina-bert-v2" },
    { LLM_ARCH_BLOOM,           "bloom"        },
    { LLM_ARCH_STABLELM,        "stablelm"     },
    { LLM_ARCH_QWEN,            "qwen"         },
    { LLM_ARCH_QWEN2,           "qwen2"        },
    { LLM_ARCH_QWEN2MOE,        "qwen2moe"     },
    { LLM_ARCH_QWEN2VL,         "qwen2vl"      },
    { LLM_ARCH_QWEN3,           "qwen3"        },
    { LLM_ARCH_QWEN3MOE,        "qwen3moe"     },
    { LLM_ARCH_QWEN3NEXT,       "qwen3next"    },
    { LLM_ARCH_QWEN3VL,         "qwen3vl"      },
    { LLM_ARCH_QWEN3VLMOE,      "qwen3vlmoe"   },
    { LLM_ARCH_QWEN35MOE,       "qwen35moe"    },
    { LLM_ARCH_QWEN35,          "qwen35"       },
    { LLM_ARCH_QWEN4EXP,        "qwen4exp"     },
    { LLM_ARCH_MELLUM,          "mellum"       },
    { LLM_ARCH_PHI2,            "phi2"         },
    { LLM_ARCH_PHI3,            "phi3"         },
    { LLM_ARCH_PLAMO,           "plamo"        },
    { LLM_ARCH_CODESHELL,       "codeshell"    },
    { LLM_ARCH_ORION,           "orion"        },
    { LLM_ARCH_INTERNLM2,       "internlm2"    },
    { LLM_ARCH_MINICPM,         "minicpm"      },
    { LLM_ARCH_GEMMA,           "gemma"        },
    { LLM_ARCH_GEMMA2,          "gemma2"       },
    { LLM_ARCH_GEMMA3,          "gemma3"       },
    { LLM_ARCH_STARCODER2,      "starcoder2"   },
    { LLM_ARCH_MAMBA,           "mamba"        },
    { LLM_ARCH_XVERSE,          "xverse"       },
    { LLM_ARCH_COMMAND_R,       "command-r"    },
    { LLM_ARCH_DBRX,            "dbrx"         },
    { LLM_ARCH_OLMO,            "olmo"         },
    { LLM_ARCH_OPENELM,         "openelm"      },
    { LLM_ARCH_ARCTIC,          "arctic"       },
    { LLM_ARCH_DEEPSEEK2,       "deepseek2"    },
    { LLM_ARCH_DEEPSEEK4,       "deepseek4"    },
    { LLM_ARCH_DEEPSEEK41,      "deepseek41"   },
    { LLM_ARCH_CHATGLM,         "chatglm"      },
    { LLM_ARCH_GLM4,            "glm4"         },
    { LLM_ARCH_GLM4_MOE,        "glm4moe"      },
    { LLM_ARCH_BITNET,          "bitnet"       },
    { LLM_ARCH_BITNET_25,       "bitnet-25"    },
    { LLM_ARCH_BITNET_B158,     "bitnet-b1.58" },
    { LLM_ARCH_T5,              "t5"           },
    { LLM_ARCH_T5ENCODER,       "t5encoder"    },
    { LLM_ARCH_JAIS,            "jais"         },
    { LLM_ARCH_GRANITE,         "granite"      },
    { LLM_ARCH_GRANITE_MOE,     "granitemoe"   },
    { LLM_ARCH_GRANITE_HYBRID,  "granitehybrid" },
    { LLM_ARCH_COHERE2,         "cohere2"      },
    { LLM_ARCH_COHERE2_MOE,     "cohere2_moe"  },
    { LLM_ARCH_DOTS1,           "dots1"        },
    { LLM_ARCH_ERNIE4_5,        "ernie4_5"     },
    { LLM_ARCH_ERNIE4_5_MOE,    "ernie4_5-moe" },
    { LLM_ARCH_HUNYUAN_MOE,     "hunyuan-moe"  },
    { LLM_ARCH_OPENAI_MOE,      "gpt-oss"      },
    { LLM_ARCH_BAILINGMOE2,     "bailingmoe2"  },
    { LLM_ARCH_BAILINGMOE3,     "bailingmoe3"  },
    { LLM_ARCH_KIMI_K3,         "kimi-k3"      },
    { LLM_ARCH_LFM2,            "lfm2"         },
    { LLM_ARCH_MINIMAX_M2,      "minimax-m2"   },
    { LLM_ARCH_MINIMAX_M3,      "minimax-m3"   },
    { LLM_ARCH_SMOLLM3,         "smollm3"      },
    { LLM_ARCH_MISTRAL3,        "mistral3"     },
    { LLM_ARCH_MIMO2,           "mimo2"        },
    { LLM_ARCH_SEED_OSS,        "seed_oss"     },
    { LLM_ARCH_STEP35,          "step35"       },
    { LLM_ARCH_LAGUNA,          "laguna"       },
    { LLM_ARCH_GLM_DSA,         "glm-dsa"      },
    { LLM_ARCH_GLM5NEXT,        "glm5next"     },
    { LLM_ARCH_GLM5NEXT_DASHED, "glm5-next"    },
    { LLM_ARCH_MISTRAL4,        "mistral4"     },
    { LLM_ARCH_GEMMA4,          "gemma4"       },
    { LLM_ARCH_DIFFUSION_GEMMA, "diffusion-gemma" },
    { LLM_ARCH_GEMMA4_MTP,      "gemma4_mtp"   },
    { LLM_ARCH_DFLASH,          "dflash"       },
    { LLM_ARCH_DFLASH2,         "dflash"       },
    { LLM_ARCH_DFLASH_DRAFT,    "dflash-draft" },
    { LLM_ARCH_GEMMA4_ASSISTANT,"gemma4-assistant"   },
    { LLM_ARCH_OPENPANGU,       "openpangu"    },
    { LLM_ARCH_MUSE_GLIMMER,    "muse-glimmer" },
    { LLM_ARCH_EAGLE3,          "eagle3"       },
    { LLM_ARCH_MODERNBERT,      "modernbert"   },
    { LLM_ARCH_UNKNOWN,         "(unknown)"    },
    { LLM_ARCH_CLIP,             "clip"             }, // dummy, only used by llama-quantize
    { LLM_ARCH_MODERN_BERT,      "modern-bert"      },
    { LLM_ARCH_NOMIC_BERT_MOE,   "nomic-bert-moe"   },
    { LLM_ARCH_NEO_BERT,         "neo-bert"         },
    { LLM_ARCH_JINA_BERT_V3,     "jina-bert-v3"     },
    { LLM_ARCH_EUROBERT,         "eurobert"         },
    { LLM_ARCH_SENSENOVA_U1,     "sensenova_u1"     },
    { LLM_ARCH_FLINT,            "flint"            },
    { LLM_ARCH_CLEF,             "clef"             },
    { LLM_ARCH_VLLM_DECISION2,   "vllm-decision2"   },
    { LLM_ARCH_PHIMOE,           "phimoe"           },
    { LLM_ARCH_PLAMO2,           "plamo2"           },
    { LLM_ARCH_PLAMO3,           "plamo3"           },
    { LLM_ARCH_MINICPM3,         "minicpm3"         },
    { LLM_ARCH_GEMMA3N,          "gemma3n"          },
    { LLM_ARCH_GEMMA_EMBEDDING,  "gemma-embedding"  },
    { LLM_ARCH_GEMMA_EMBEDDING2, "gemma-embedding2" },
    { LLM_ARCH_MAMBA2,           "mamba2"           },
    { LLM_ARCH_MAPLE,            "maple"            },
    { LLM_ARCH_JAMBA,            "jamba"            },
    { LLM_ARCH_FALCON_H1,        "falcon-h1"        },
    { LLM_ARCH_COHERE2MOE,       "cohere2moe"       },
    { LLM_ARCH_OLMO2,            "olmo2"            },
    { LLM_ARCH_OLMOE,            "olmoe"            },
    { LLM_ARCH_MUSE_SPARK,       "muse-spark"       },
    { LLM_ARCH_DEEPSEEK,         "deepseek"         },
    { LLM_ARCH_DEEPSEEK2OCR,     "deepseek2-ocr"    },
    { LLM_ARCH_DEEPSEEK32,       "deepseek32"       },
    { LLM_ARCH_JAIS2,            "jais2"            },
    { LLM_ARCH_NEMOTRON,         "nemotron"         },
    { LLM_ARCH_NEMOTRON_H,       "nemotron_h"       },
    { LLM_ARCH_NEMOTRON_H_MOE,   "nemotron_h_moe"   },
    { LLM_ARCH_EXAONE,           "exaone"           },
    { LLM_ARCH_EXAONE4,          "exaone4"          },
    { LLM_ARCH_EXAONE_MOE,       "exaone-moe"       },
    { LLM_ARCH_RWKV6,            "rwkv6"            },
    { LLM_ARCH_RWKV6QWEN2,       "rwkv6qwen2"       },
    { LLM_ARCH_RWKV7,            "rwkv7"            },
    { LLM_ARCH_ARWKV7,           "arwkv7"           },
    { LLM_ARCH_GRANITE_SWITCH,   "graniteswitch"    },
    { LLM_ARCH_GRANITE_SWA,      "granite_swa"      },
    { LLM_ARCH_CHAMELEON,        "chameleon"        },
    { LLM_ARCH_WAVTOKENIZER_DEC, "wavtokenizer-dec" },
    { LLM_ARCH_PLM,              "plm"              },
    { LLM_ARCH_BAILINGMOE,       "bailingmoe"       },
    { LLM_ARCH_DOTS3NOTE,        "dots3note"        },
    { LLM_ARCH_ARCEE,            "arcee"            },
    { LLM_ARCH_AFMOE,            "afmoe"            },
    { LLM_ARCH_HUNYUAN_DENSE,    "hunyuan-dense"    },
    { LLM_ARCH_HUNYUAN_VL,       "hunyuan_vl"       },
    { LLM_ARCH_HY_V3,            "hy_v3"            },
    { LLM_ARCH_HY_V4,             "hy_v4" },
    { LLM_ARCH_SPARK2_5,          "spark2_5" },
    { LLM_ARCH_LFM2MOE,          "lfm2moe"          },
    { LLM_ARCH_DREAM,            "dream"            },
    { LLM_ARCH_SMALLTHINKER,     "smallthinker"     },
    { LLM_ARCH_LLADA,            "llada"            },
    { LLM_ARCH_LLADA_MOE,        "llada-moe"        },
    { LLM_ARCH_GROVEMOE,         "grovemoe"         },
    { LLM_ARCH_APERTUS,          "apertus"          },
    { LLM_ARCH_MINIMAX_01,       "minimax-01"       },
    { LLM_ARCH_HRM_TEXT,         "hrm_text"         },
    { LLM_ARCH_COGVLM,           "cogvlm"           },
    { LLM_ARCH_RND1,             "rnd1"             },
    { LLM_ARCH_PANGU_EMBED,      "pangu-embedded"   },
    { LLM_ARCH_PADDLEOCR,        "paddleocr"        },
    { LLM_ARCH_LLAMA_EMBED,      "llama-embed"      },
    { LLM_ARCH_MAINCODER,        "maincoder"        },
    { LLM_ARCH_MAION_CODER,      "maion-coder"      },
    { LLM_ARCH_KIMI_LINEAR,      "kimi-linear"      },
    { LLM_ARCH_GLM5_NEXT,        "glm5-next"        },
    { LLM_ARCH_TALKIE,           "talkie"           },
    { LLM_ARCH_NANBEIGE,         "nanbeige"         },
    { LLM_ARCH_QWEN3TTS,         "qwen3tts"         },
    { LLM_ARCH_POCKETTTS,        "pockettts"        },
    { LLM_ARCH_INKLING,          "inkling"          },
    { LLM_ARCH_HYPERCLOVAX,      "hyperclovax"      },
    { LLM_ARCH_BEAM,             "beam"             },
    { LLM_ARCH_KOLIBRI1,         "kolibri1"         },
    { LLM_ARCH_MMFREELM,         "mmfreelm"         },
};

// S69: architecture-name ALIASES for GGUFs exported by mainline llama.cpp.
//
// LLM_ARCH_NAMES maps our enum to the string matched against a GGUF's general.architecture.
// Mainline (ggml-org/llama.cpp) spells two architectures we already implement differently, so a
// model exported by their tooling fails here as "unknown architecture" while every source-level
// check confirms the architecture IS implemented. That is the worst kind of failure, because
// grepping the codebase confirms the wrong answer.
//
// MEASURED 2026-08-20 by parsing LLM_ARCH_NAMES on both sides and comparing them normalised (with
// a present-control and a fabricated-control): of 65 names present in mainline and literally
// absent here, exactly 2 are spelling divergences and 63 are genuine gaps. cohere2moe is
// Command A+ MoE, which is on our benchmark list.
//
// Consulted ONLY after the exact-match scan fails, so our own GGUFs -- which declare our spelling
// -- are unaffected. Additive by construction: nothing is renamed, so no file that loads today
// stops loading. Extend this table rather than editing LLM_ARCH_NAMES, which would invalidate
// every GGUF we have already produced.
static const std::map<std::string, llm_arch> LLM_ARCH_NAME_ALIASES = {
    { "cohere2moe",  LLM_ARCH_COHERE2_MOE },   // ours: "cohere2_moe"
    { "modern-bert", LLM_ARCH_MODERNBERT  },   // ours: "modernbert"
};

llm_arch llm_arch_from_string(const std::string & name) {
    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    if (auto it = LLM_ARCH_NAME_ALIASES.find(name); it != LLM_ARCH_NAME_ALIASES.end()) {
        return it->second;
    }

    return llm_arch_from_ollama_name(name);
}

// Architecture names Ollama's converter writes for models whose TEXT part is one of ours unchanged.
// Each entry was read in Ollama's source, not inferred from the name:
//   qwen25vl -- convert/convert_qwen25vl.go:34-40 copies every qwen2.* key to qwen25vl.* and keeps the
//               qwen2 tensor names; model/models/qwen25vl/model_text.go:20-22,69-89 is Qwen2 + M-RoPE
//               (rope.mrope_section), i.e. our qwen2vl text graph. The vision tower (v.*) in the same
//               file is left to the combined-VL handling in done_getting_tensors.
// The loader reads keys under the FILE's prefix (LLM_KV::prefix) for any aliased name.
llm_arch llm_arch_from_ollama_name(const std::string & name) {
    static const std::map<std::string, llm_arch> k_ollama = {
        { "qwen25vl", LLM_ARCH_QWEN2VL },
    };
    auto it = k_ollama.find(name);
    return it != k_ollama.end() ? it->second : LLM_ARCH_UNKNOWN;
}

const char * llama_model_arch_string(const struct llama_model * model) {
    static const char * unknown = "unknown";
    if (!model) return unknown;
    if (auto it = LLM_ARCH_NAMES.find(model->arch); it != LLM_ARCH_NAMES.end()) {
        return it->second;
    }
    return unknown;
}

static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_TYPE,                  "general.type"                          },
    { LLM_KV_GENERAL_ARCHITECTURE,          "general.architecture"                  },
    { LLM_KV_GENERAL_QUANTIZATION_VERSION,  "general.quantization_version"          },
    { LLM_KV_GENERAL_ALIGNMENT,             "general.alignment"                     },
    { LLM_KV_GENERAL_NAME,                  "general.name"                          },
    { LLM_KV_GENERAL_AUTHOR,                "general.author"                        },
    { LLM_KV_GENERAL_VERSION,               "general.version"                       },
    { LLM_KV_GENERAL_URL,                   "general.url"                           },
    { LLM_KV_GENERAL_DESCRIPTION,           "general.description"                   },
    { LLM_KV_GENERAL_LICENSE,               "general.license"                       },
    { LLM_KV_GENERAL_SOURCE_URL,            "general.source.url"                    },
    { LLM_KV_GENERAL_SOURCE_HF_REPO,        "general.source.huggingface.repository" },
    { LLM_KV_GENERAL_TENSOR_EXTRA_NAME,              "general.tensor_extra.name"             },
    { LLM_KV_GENERAL_TENSOR_EXTRA_PREC_A4,           "general.tensor_extra.prec_a4"          },

    { LLM_KV_VOCAB_SIZE,                        "%s.vocab_size"                        },
    { LLM_KV_CONTEXT_LENGTH,                    "%s.context_length"                    },
    { LLM_KV_EMBEDDING_LENGTH,                  "%s.embedding_length"                  },
    { LLM_KV_EMBEDDING_LENGTH_OUT,              "%s.embedding_length_out"              },
    { LLM_KV_EMBEDDING_LENGTH_PER_LAYER,        "%s.embedding_length_per_layer_input"  },
    { LLM_KV_BLOCK_COUNT,                       "%s.block_count"                       },
    { LLM_KV_LEADING_DENSE_BLOCK_COUNT,         "%s.leading_dense_block_count"         },
    { LLM_KV_FEED_FORWARD_LENGTH,               "%s.feed_forward_length"               },
    { LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        "%s.expert_feed_forward_length"        },
    { LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, "%s.expert_shared_feed_forward_length" },
    { LLM_KV_USE_PARALLEL_RESIDUAL,             "%s.use_parallel_residual"             },
    { LLM_KV_TENSOR_DATA_LAYOUT,                "%s.tensor_data_layout"                },
    { LLM_KV_EXPERT_COUNT,                      "%s.expert_count"                      },
    { LLM_KV_EXPERT_USED_COUNT,                 "%s.expert_used_count"                 },
    { LLM_KV_EXPERT_SHARED_COUNT,               "%s.expert_shared_count"               },
    { LLM_KV_EXPERT_GROUP_COUNT,                "%s.expert_group_count"                },
    { LLM_KV_EXPERT_GROUP_USED_COUNT,           "%s.expert_group_used_count"           },
    { LLM_KV_EXPERT_WEIGHTS_SCALE,              "%s.expert_weights_scale"              },
    { LLM_KV_EXPERT_WEIGHTS_NORM,               "%s.expert_weights_norm"               },
    { LLM_KV_SHORTCONV_L_CACHE,                 "%s.shortconv.l_cache"                 },
    { LLM_KV_EXPERT_LATENT_LENGTH,               "%s.expert_latent_length"              },
    { LLM_KV_ACTIVATION_SITU_BETA,               "%s.activation.situ_beta"              },
    { LLM_KV_ACTIVATION_SITU_LINEAR_BETA,        "%s.activation.situ_linear_beta"       },
    { LLM_KV_ATTN_RES_BLOCK_SIZE,                "%s.attn_res.block_size"               },
    { LLM_KV_EXPERT_GATING_FUNC,                "%s.expert_gating_func"                },
    // K2 Horizon MoVA
    { LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,     "%s.attention.value_expert_count"       },
    { LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT,"%s.attention.value_expert_used_count"  },

    { LLM_KV_POST_NORM_EPS,                     "%s.post_norm_epsilon"                 },
    { LLM_KV_NEXTN_PREDICT_LAYERS,              "%s.nextn_predict_layers"              },
    { LLM_KV_NUM_DEEPSTACK_LAYERS,              "%s.n_deepstack_layers"                },
    { LLM_KV_POOLING_TYPE,                      "%s.pooling_type"                      },
    { LLM_KV_LOGIT_SCALE,                       "%s.logit_scale"                       },
    { LLM_KV_DECODER_START_TOKEN_ID,            "%s.decoder_start_token_id"            },
    { LLM_KV_ATTN_LOGIT_SOFTCAPPING,            "%s.attn_logit_softcapping"            },
    { LLM_KV_ROUTER_LOGIT_SOFTCAPPING,          "%s.router_logit_softcapping"          },
    { LLM_KV_FINAL_LOGIT_SOFTCAPPING,           "%s.final_logit_softcapping"           },
    { LLM_KV_RESIDUAL_SCALE,                    "%s.residual_scale"                    },
    { LLM_KV_EMBEDDING_SCALE,                   "%s.embedding_scale"                   },
    { LLM_KV_TOKEN_SHIFT_COUNT,                 "%s.token_shift_count"                 },
    { LLM_KV_INTERLEAVE_MOE_LAYER_STEP,         "%s.interleave_moe_layer_step"         },
    { LLM_KV_SWIGLU_LIMITS,                     "%s.swiglu_limits"                     },
    { LLM_KV_SWIGLU_LIMITS_SHARED,              "%s.swiglu_limits_shared"              },
    { LLM_KV_SWIGLU_CLAMP_EXP,                  "%s.swiglu_clamp_exp"                  },
    { LLM_KV_SWIGLU_CLAMP_SHEXP,                "%s.swiglu_clamp_shexp"                },
    { LLM_KV_MTP_BACKBONE_EMBEDDING_LENGTH,     "%s.backbone_embedding_length"         },
    { LLM_KV_MTP_USE_ORDERED_EMBEDDINGS,        "%s.use_ordered_embeddings"            },
    { LLM_KV_MTP_CENTROID_COUNT,                "%s.centroid_count"                    },
    { LLM_KV_MTP_CENTROID_TOP_K,                "%s.centroid_top_k"                    },
    { LLM_KV_DFLASH_BLOCK_SIZE,                 "%s.dflash.block_size"                 },
    { LLM_KV_DFLASH_MASK_TOKEN_ID,              "%s.dflash.mask_token_id"              },
    { LLM_KV_DFLASH_TARGET_LAYER_IDS,           "%s.dflash.target_layer_ids"           },
    { LLM_KV_DFLASH_TARGET_LAYERS,              "%s.target_layers"                      },
    { LLM_KV_DFLASH_N_TARGET_FEATURES,          "%s.dflash.n_target_features"          },
    { LLM_KV_DFLASH_BACKBONE_ROTARY_BASE,       "%s.dflash.backbone_rotary_base"       },
    { LLM_KV_DFLASH_LAGUNA,                     "%s.dflash.laguna"                     },
    { LLM_KV_DFLASH_CONV_KERNEL_SIZE,            "%s.conv_kernel_size"                   },
    { LLM_KV_DFLASH_CONV_GROUP_SIZE,             "%s.conv_group_size"                    },
    { LLM_KV_DFLASH_SELECTOR_RANK,               "%s.selector_rank"                      },
    // upstream b10f9ca58 spells this "{arch}.selector_top_k" (NOT nested under .dflash.), and that
    // is the only spelling any converter actually writes, so it is kept verbatim. ik_llama #2345 uses the same spelling.
    { LLM_KV_DFLASH_SELECTOR_TOP_K,              "%s.selector_top_k"                     },

    { LLM_KV_ATTENTION_HEAD_COUNT,             "%s.attention.head_count"             },
    { LLM_KV_ATTENTION_HEAD_COUNT_KV,          "%s.attention.head_count_kv"          },
    { LLM_KV_ATTENTION_MAX_ALIBI_BIAS,         "%s.attention.max_alibi_bias"         },
    { LLM_KV_ATTENTION_CLAMP_KQV,              "%s.attention.clamp_kqv"              },
    { LLM_KV_ATTENTION_KEY_LENGTH,             "%s.attention.key_length"             },
    { LLM_KV_ATTENTION_VALUE_LENGTH,           "%s.attention.value_length"           },
    { LLM_KV_ATTENTION_LAYERNORM_EPS,          "%s.attention.layer_norm_epsilon"     },
    { LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      "%s.attention.layer_norm_rms_epsilon" },
    { LLM_KV_ATTENTION_CAUSAL,                 "%s.attention.causal"                 },
    // S69 #5/#30 — EAGLE3 draft-head metadata (PR #18039 port; conversion #29 emits these keys)
    { LLM_KV_TARGET_LAYERS,                    "%s.target_layers"                    },
    { LLM_KV_TARGET_HIDDEN_SIZE,               "%s.target_hidden_size"               },
    { LLM_KV_NORM_BEFORE_RESIDUAL,             "%s.norm_before_residual"             },
    { LLM_KV_DRAFT_VOCAB_SIZE,                 "%s.draft_vocab_size"                 },
    { LLM_KV_ATTENTION_Q_LORA_RANK,            "%s.attention.q_lora_rank"            },
    { LLM_KV_ATTENTION_KV_LORA_RANK,           "%s.attention.kv_lora_rank"           },
    { LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, "%s.attention.relative_buckets_count" },
    { LLM_KV_ATTENTION_SLIDING_WINDOW,         "%s.attention.sliding_window"         },
    { LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, "%s.attention.sliding_window_pattern" },
    { LLM_KV_ATTENTION_SCALE,                  "%s.attention.scale"                  },
    { LLM_KV_ATTENTION_OUTPUT_SCALE,           "%s.attention.output_scale"           },
    { LLM_KV_ATTENTION_TEMPERATURE_LENGTH,     "%s.attention.temperature_length"     },
    { LLM_KV_ATTENTION_TEMPERATURE_SCALE,      "%s.attention.temperature_scale"      },
    { LLM_KV_ATTENTION_KEY_LENGTH_MLA,         "%s.attention.key_length_mla"         },
    { LLM_KV_ATTENTION_VALUE_LENGTH_MLA,       "%s.attention.value_length_mla"       },
    { LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,     "%s.attention.indexer.head_count"     },
    { LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,     "%s.attention.indexer.key_length"     },
    { LLM_KV_ATTENTION_INDEXER_TOP_K,          "%s.attention.indexer.top_k"          },
    { LLM_KV_ATTENTION_INDEXER_KPOOL,          "%s.attention.indexer.kpool"          },
    { LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,     "%s.attention.output_group_count"     },
    { LLM_KV_ATTENTION_OUTPUT_LORA_RANK,       "%s.attention.output_lora_rank"       },
    { LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,"%s.attention.compress_rope_freq_base"},
    { LLM_KV_ATTENTION_COMPRESS_RATIOS,        "%s.attention.compress_ratios"        },
    { LLM_KV_FULL_ATTENTION_INTERVAL,          "%s.full_attention_interval"          },
    { LLM_KV_ATTENTION_SHARED_KV_LAYERS,       "%s.attention.shared_kv_layers"       },
    { LLM_KV_ATTENTION_KEY_LENGTH_SWA,         "%s.attention.key_length_swa"         },
    { LLM_KV_ATTENTION_VALUE_LENGTH_SWA,       "%s.attention.value_length_swa"       },
    { LLM_KV_ATTENTION_VALUE_SCALE,            "%s.attention.value_scale"            },

    { LLM_KV_HYPER_CONNECTION_COUNT,               "%s.hyper_connection.count"               },
    { LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, "%s.hyper_connection.sinkhorn_iterations" },
    { LLM_KV_HYPER_CONNECTION_EPSILON,             "%s.hyper_connection.epsilon"             },
    { LLM_KV_HYPER_CONNECTION_MAGNITUDE,            "%s.hyper_connection.magnitude" },
    { LLM_KV_HYPER_CONNECTION_LOW_RANK,            "%s.hyper_connection.low_rank"            },

    { LLM_KV_PLE_LAYERS,                       "%s.ple.layers"                      },
    { LLM_KV_PLE_NGRAM_SIZE,                   "%s.ple.ngram_size"                  },
    { LLM_KV_PLE_HEADS_PER_NGRAM,              "%s.ple.heads_per_ngram"             },
    { LLM_KV_PLE_CONV_KERNEL,                  "%s.ple.conv_kernel"                 },
    { LLM_KV_PLE_LAYER_MULTIPLIERS,            "%s.ple.layer_multipliers"           },
    { LLM_KV_PLE_HEAD_OFFSETS,                 "%s.ple.head_offsets"                },
    { LLM_KV_PLE_HEAD_VOCAB_SIZES,             "%s.ple.head_vocab_sizes"            },
    { LLM_KV_PLE_EOS_TOKEN_ID,                 "%s.ple.eos_token_id"                },
    { LLM_KV_PLE_IMAGE_TOKEN_ID,               "%s.ple.image_token_id"              },

    { LLM_KV_HASH_LAYER_COUNT,                 "%s.hash_layer_count"                },
    { LLM_KV_ENGRAM_HEAD_COUNT,      "%s.engram.head_count" },
    { LLM_KV_ENGRAM_KEY_LENGTH,      "%s.engram.key_length" },
    { LLM_KV_ENGRAM_MAX_NGRAM_SIZE,  "%s.engram.max_ngram_size" },
    { LLM_KV_ENGRAM_LAYER_IDS,       "%s.engram.layer_ids" },
    { LLM_KV_ENGRAM_MULTIPLIERS,     "%s.engram.multipliers" },
    { LLM_KV_ENGRAM_PRIMES,          "%s.engram.primes" },
    { LLM_KV_ENGRAM_OFFSETS,         "%s.engram.offsets" },
    { LLM_KV_ENGRAM_TOKEN_MAP,       "%s.engram.token_map" },
    { LLM_KV_ENGRAM_PAD_ID,          "%s.engram.pad_id" },
    { LLM_KV_CANDIDATE_SOURCE_LAYER,         "%s.attention.candidate_source_layer_id" },
    { LLM_KV_CANDIDATE_BLOCK_SIZE,           "%s.attention.candidate_block_size"      },
    { LLM_KV_CANDIDATE_TOPK_BLOCKS,          "%s.attention.candidate_topk_blocks"     },

    { LLM_KV_ROPE_DIMENSION_COUNT,          "%s.rope.dimension_count"                 },
    { LLM_KV_ROPE_DIMENSION_COUNT_SWA,      "%s.rope.dimension_count_swa"             },
    { LLM_KV_ROPE_DIMENSION_COUNT_PER_LAYER,"%s.rope.dimension_count_per_layer"       },
    { LLM_KV_ROPE_DIMENSION_SECTIONS,       "%s.rope.dimension_sections"              },
    { LLM_KV_ROPE_FREQ_BASE,                "%s.rope.freq_base"                       },
    { LLM_KV_ROPE_FREQ_BASE_PER_LAYER,      "%s.rope.freq_base_per_layer"             },
    { LLM_KV_ROPE_FREQ_BASE_SWA,            "%s.rope.freq_base_swa"                   },
    { LLM_KV_ROPE_SCALE_LINEAR,             "%s.rope.scale_linear"                    },
    { LLM_KV_ROPE_SCALING_TYPE,             "%s.rope.scaling.type"                    },
    { LLM_KV_ROPE_SCALING_FACTOR,           "%s.rope.scaling.factor"                  },
    { LLM_KV_ROPE_SCALING_ATTN_FACTOR,      "%s.rope.scaling.attn_factor"             },
    { LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,     "%s.rope.scaling.original_context_length" },
    { LLM_KV_ROPE_SCALING_FINETUNED,        "%s.rope.scaling.finetuned"               },
    { LLM_KV_ROPE_SCALING_YARN_LOG_MUL,     "%s.rope.scaling.yarn_log_multiplier"     },
    { LLM_KV_ROPE_SCALING_YARN_MSCALE,      "%s.rope.scaling.yarn_mscale"             },
    { LLM_KV_ROPE_SCALING_YARN_TRUNCATE,    "%s.rope.scaling.yarn_truncate"           },
    { LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR,  "%s.rope.scaling.yarn_ext_factor"         },
    { LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, "%s.rope.scaling.yarn_attn_factor"        },
    { LLM_KV_ROPE_SCALING_YARN_BETA_FAST,   "%s.rope.scaling.yarn_beta_fast"          },
    { LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   "%s.rope.scaling.yarn_beta_slow"          },

    { LLM_KV_SPLIT_NO,                      "split.no"            },
    { LLM_KV_SPLIT_COUNT,                   "split.count"         },
    { LLM_KV_SPLIT_TENSORS_COUNT,           "split.tensors.count" },

    { LLM_KV_SSM_CONV_KERNEL,               "%s.ssm.conv_kernel"    },
    { LLM_KV_SSM_INNER_SIZE,                "%s.ssm.inner_size"     },
    { LLM_KV_SSM_STATE_SIZE,                "%s.ssm.state_size"     },
    { LLM_KV_SSM_TIME_STEP_RANK,            "%s.ssm.time_step_rank" },
    { LLM_KV_SSM_GROUP_COUNT,               "%s.ssm.group_count"    },
    { LLM_KV_KDA_HEAD_DIM,                  "%s.kda.head_dim"       },
    { LLM_KV_KDA_SAFE_GATE,                 "%s.kda.safe_gate"      },
    { LLM_KV_KDA_GATE_LOWER_BOUND,          "%s.kda.gate_lower_bound" },

    { LLM_KV_TOKENIZER_MODEL,                "tokenizer.ggml.model"                    },
    { LLM_KV_TOKENIZER_PRE,                  "tokenizer.ggml.pre"                      },
    { LLM_KV_TOKENIZER_LIST,                 "tokenizer.ggml.tokens"                   },
    { LLM_KV_TOKENIZER_TOKEN_TYPE,           "tokenizer.ggml.token_type"               },
    { LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,     "tokenizer.ggml.token_type_count"         },
    { LLM_KV_TOKENIZER_SCORES,               "tokenizer.ggml.scores"                   },
    { LLM_KV_TOKENIZER_MERGES,               "tokenizer.ggml.merges"                   },
    { LLM_KV_TOKENIZER_BOS_ID,               "tokenizer.ggml.bos_token_id"             },
    { LLM_KV_TOKENIZER_EOS_ID,               "tokenizer.ggml.eos_token_id"             },
    { LLM_KV_TOKENIZER_UNK_ID,               "tokenizer.ggml.unknown_token_id"         },
    { LLM_KV_TOKENIZER_SEP_ID,               "tokenizer.ggml.seperator_token_id"       },
    { LLM_KV_TOKENIZER_PAD_ID,               "tokenizer.ggml.padding_token_id"         },
    { LLM_KV_TOKENIZER_CLS_ID,               "tokenizer.ggml.cls_token_id"             },
    { LLM_KV_TOKENIZER_MASK_ID,              "tokenizer.ggml.mask_token_id"            },
    { LLM_KV_TOKENIZER_ADD_BOS,              "tokenizer.ggml.add_bos_token"            },
    { LLM_KV_TOKENIZER_ADD_EOS,              "tokenizer.ggml.add_eos_token"            },
    { LLM_KV_TOKENIZER_ADD_SEP,              "tokenizer.ggml.add_sep_token"            },
    { LLM_KV_TOKENIZER_ADD_PREFIX,           "tokenizer.ggml.add_space_prefix"         },
    { LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,      "tokenizer.ggml.remove_extra_whitespaces" },
    { LLM_KV_TOKENIZER_DO_LOWER_CASE,        "tokenizer.ggml.do_lower_case"            },
    { LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP, "tokenizer.ggml.precompiled_charsmap"     },
    { LLM_KV_TOKENIZER_HF_JSON,              "tokenizer.huggingface.json"              },
    { LLM_KV_TOKENIZER_RWKV,                 "tokenizer.rwkv.world"                    },
    { LLM_KV_TOKENIZER_CHAT_TEMPLATE,        "tokenizer.chat_template"                 },
    { LLM_KV_TOKENIZER_CHAT_TEMPLATE_N,      "tokenizer.chat_template.%s"              },
    { LLM_KV_TOKENIZER_FIM_PRE_ID,           "tokenizer.ggml.fim_pre_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SUF_ID,           "tokenizer.ggml.fim_suf_token_id"         },
    { LLM_KV_TOKENIZER_FIM_MID_ID,           "tokenizer.ggml.fim_mid_token_id"         },
    { LLM_KV_TOKENIZER_FIM_PAD_ID,           "tokenizer.ggml.fim_pad_token_id"         },
    { LLM_KV_TOKENIZER_FIM_REP_ID,           "tokenizer.ggml.fim_rep_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SEP_ID,           "tokenizer.ggml.fim_sep_token_id"         },

    { LLM_KV_TOKENIZER_PREFIX_ID,            "tokenizer.ggml.prefix_token_id"          },
    { LLM_KV_TOKENIZER_SUFFIX_ID,            "tokenizer.ggml.suffix_token_id"          },
    { LLM_KV_TOKENIZER_MIDDLE_ID,            "tokenizer.ggml.middle_token_id"          },
    { LLM_KV_TOKENIZER_EOT_ID,               "tokenizer.ggml.eot_token_id"             },
    { LLM_KV_TOKENIZER_EOM_ID,               "tokenizer.ggml.eom_token_id"             },

    { LLM_KV_ADAPTER_TYPE,                  "adapter.type"       },
    { LLM_KV_ADAPTER_LORA_ALPHA,            "adapter.lora.alpha" },

    { LLM_KV_OPENPANGU_MHC_NUM_STREAM,      "%s.mhc_num_stream"     },
    { LLM_KV_OPENPANGU_MHC_RECUR_NORM,      "%s.mhc_recur_norm"     },
    { LLM_KV_OPENPANGU_PARAM_SINK_NUMBER,   "%s.param_sink_number"  },
    { LLM_KV_GENERAL_FILE_TYPE,                "general.file_type"                     },
    { LLM_KV_GENERAL_SAMPLING_SEQUENCE,        "general.sampling.sequence"             },
    { LLM_KV_GENERAL_SAMPLING_TOP_K,           "general.sampling.top_k"                },
    { LLM_KV_GENERAL_SAMPLING_TOP_P,           "general.sampling.top_p"                },
    { LLM_KV_GENERAL_SAMPLING_MIN_P,           "general.sampling.min_p"                },
    { LLM_KV_GENERAL_SAMPLING_XTC_PROBABILITY, "general.sampling.xtc_probability"      },
    { LLM_KV_GENERAL_SAMPLING_XTC_THRESHOLD,   "general.sampling.xtc_threshold"        },
    { LLM_KV_GENERAL_SAMPLING_TEMP,            "general.sampling.temp"                 },
    { LLM_KV_GENERAL_SAMPLING_PENALTY_LAST_N,  "general.sampling.penalty_last_n"       },
    { LLM_KV_GENERAL_SAMPLING_PENALTY_REPEAT,  "general.sampling.penalty_repeat"       },
    { LLM_KV_GENERAL_SAMPLING_MIROSTAT,        "general.sampling.mirostat"             },
    { LLM_KV_GENERAL_SAMPLING_MIROSTAT_TAU,    "general.sampling.mirostat_tau"         },
    { LLM_KV_GENERAL_SAMPLING_MIROSTAT_ETA,    "general.sampling.mirostat_eta"         },
    { LLM_KV_FEATURES_LENGTH,                   "%s.features_length"                   },
    { LLM_KV_EXPERT_CHUNK_FEED_FORWARD_LENGTH,  "%s.expert_chunk_feed_forward_length"  },
    { LLM_KV_EXPERT_GROUP_SCALE,                "%s.expert_group_scale"                },
    { LLM_KV_EXPERTS_PER_GROUP,                 "%s.experts_per_group"                 },
    { LLM_KV_MOE_EVERY_N_LAYERS,                "%s.moe_every_n_layers"                },
    { LLM_KV_MOE_LATENT_SIZE,                   "%s.moe_latent_size"                   },
    { LLM_KV_DEEPSTACK_MAPPING,                 "%s.deepstack_mapping"                 },
    { LLM_KV_HIDDEN_ACT,                        "%s.hidden_activation"                 },
    { LLM_KV_DECODER_BLOCK_COUNT,               "%s.decoder_block_count"               },
    { LLM_KV_SWIN_NORM,                         "%s.swin_norm"                         },
    { LLM_KV_RESCALE_EVERY_N_LAYERS,            "%s.rescale_every_n_layers"            },
    { LLM_KV_TIME_MIX_EXTRA_DIM,                "%s.time_mix_extra_dim"                },
    { LLM_KV_TIME_DECAY_EXTRA_DIM,              "%s.time_decay_extra_dim"              },
    { LLM_KV_ADAPTER_COUNT,                     "%s.adapters.count"                    },
    { LLM_KV_ADAPTER_TOKEN_IDS_ACTIVATE,        "%s.adapters.token_ids_activate"       },
    { LLM_KV_ADAPTER_TOKEN_IDS_SUBSTITUTE,      "%s.adapters.token_ids_substitute"     },
    { LLM_KV_ADAPTER_LORA_RANK,                 "%s.adapters.lora_rank"                },
    { LLM_KV_ADAPTER_ROUTER_GAIN,               "%s.adapters.router_gain"              },
    { LLM_KV_NUM_LOOPS,                         "%s.num_loops"                         },
    { LLM_KV_SKIP_LOOP_FINAL_NORM,              "%s.skip_loop_final_norm"              },
    { LLM_KV_HRM_LAYERS_PER_STACK,              "%s.hrm.layers_per_stack"              },
    { LLM_KV_HRM_H_CYCLES,                      "%s.hrm.h_cycles"                      },
    { LLM_KV_HRM_L_CYCLES,                      "%s.hrm.l_cycles"                      },
    { LLM_KV_HRM_PREFIX_LM,                     "%s.hrm.prefix_lm"                     },
    { LLM_KV_ATTENTION_GROUPNORM_EPS,                "%s.attention.group_norm_epsilon"           },
    { LLM_KV_ATTENTION_GROUPNORM_GROUPS,             "%s.attention.group_norm_groups"            },
    { LLM_KV_ATTENTION_DECAY_LORA_RANK,              "%s.attention.decay_lora_rank"              },
    { LLM_KV_ATTENTION_ICLR_LORA_RANK,               "%s.attention.iclr_lora_rank"               },
    { LLM_KV_ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK, "%s.attention.value_residual_mix_lora_rank" },
    { LLM_KV_ATTENTION_GATE_LORA_RANK,               "%s.attention.gate_lora_rank"               },
    { LLM_KV_ATTENTION_ROPE_PATTERN,                 "%s.attention.rope_pattern"                 },
    { LLM_KV_REALTIME_DELAY_TOKENS,                  "%s.realtime_delay_tokens"                  },
    { LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,           "%s.attention.key_length_mla_swa"           },
    { LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA,         "%s.attention.value_length_mla_swa"         },
    { LLM_KV_ATTENTION_KV_LORA_RANK_SWA,             "%s.attention.kv_lora_rank_swa"             },
    { LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,           "%s.attention.indexer.block_size"           },
    { LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,         "%s.attention.indexer.local_blocks"         },
    { LLM_KV_ATTENTION_INDEXER_TYPES,                "%s.attention.indexer.types"                },
    { LLM_KV_ATTENTION_RECURRENT_LAYERS,             "%s.attention.recurrent_layers"             },
    { LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL,    "%s.attention.indexer.kpool_select_tail"    },
    { LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS,          "%s.attention.indexer.source_layers"          },
    { LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, "%s.attention.indexer.candidate_source_layer" },
    { LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, "%s.attention.indexer.candidate_top_k_blocks" },
    { LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   "%s.attention.indexer.candidate_block_size"   },
    { LLM_KV_ATTENTION_KV_SOURCE_LAYERS,               "%s.attention.kv_source_layers"               },
    { LLM_KV_HYPER_CONNECTION_SINGLE_PASS,           "%s.hyper_connection.single_pass"           },
    { LLM_KV_ENGRAM_PAD_TOKEN_ID,                    "%s.engram.pad_token_id"                    },
    { LLM_KV_ROPE_SECTION_ORDER,             "%s.rope.section_order"                   },
    { LLM_KV_ROPE_FREQ_BASE_SPATIAL,         "%s.rope.freq_base_spatial"               },
    { LLM_KV_ROPE_SCALING_ALPHA,             "%s.rope.scaling.alpha"                   },
    { LLM_KV_SSM_DT_B_C_RMS,     "%s.ssm.dt_b_c_rms"     },

    { LLM_KV_WKV_HEAD_SIZE, "%s.wkv.head_size" },

    { LLM_KV_POSNET_EMBEDDING_LENGTH, "%s.posnet.embedding_length" },
    { LLM_KV_POSNET_BLOCK_COUNT,      "%s.posnet.block_count"      },

    { LLM_KV_CONVNEXT_EMBEDDING_LENGTH, "%s.convnext.embedding_length" },
    { LLM_KV_CONVNEXT_BLOCK_COUNT,      "%s.convnext.block_count"      },

    { LLM_KV_CLASSIFIER_OUTPUT_LABELS, "%s.classifier.output_labels" },
    { LLM_KV_NORM_BEFORE_FC,        "%s.norm_before_fc"       },
    { LLM_KV_CLASSIFIER_POOLING_TYPE,  "%s.classifier.pooling_type"  },
    { LLM_KV_CLASSIFIER_ACTIVATION,    "%s.classifier.activation"    },
    { LLM_KV_DECISION_BLOCK_COUNT,         "%s.decision.block_count"         },
    { LLM_KV_DECISION_ROUTING_BLOCK_COUNT, "%s.decision.routing_block_count" },
    { LLM_KV_DECISION_HEAD_COUNT,          "%s.decision.head_count"          },
    { LLM_KV_DECISION_LABELS,              "%s.decision.labels"              },
    { LLM_KV_LILICORR_HIDDEN_SIZE,       "lilicorr.hidden_size"      },
    { LLM_KV_LILICORR_CANDIDATE_TOPK,    "lilicorr.candidate_topk"   },
    { LLM_KV_LILICORR_NUM_HEADS,         "lilicorr.num_heads"        },
    { LLM_KV_LILICORR_NUM_LAYERS,        "lilicorr.num_layers"       },
    { LLM_KV_LILICORR_VECTOR_EPS,        "lilicorr.vector_eps"       },
    { LLM_KV_LILICORR_LOGIT_SCALE,       "lilicorr.logit_scale"      },
    { LLM_KV_LILICORR_FACTOR_DIM,        "lilicorr.factor_dim"       },
    { LLM_KV_LILICORR_MLP_HIDDEN_SIZE,   "lilicorr.mlp_hidden_size"  },
    // sentence-transformers dense modules feature dims
    { LLM_KV_DENSE_2_FEAT_IN,        "%s.dense_2_feat_in"  },
    { LLM_KV_DENSE_2_FEAT_OUT,       "%s.dense_2_feat_out" },
    { LLM_KV_DENSE_3_FEAT_IN,        "%s.dense_3_feat_in"  },
    { LLM_KV_DENSE_3_FEAT_OUT,       "%s.dense_3_feat_out" },
    // inkling (S69 P4). SOURCE-READ: src__llama-arch.cpp:368-377 of the saved upstream branch (MIT)
    { LLM_KV_INKLING_D_REL,               "%s.d_rel"               },
    { LLM_KV_INKLING_REL_EXTENT,          "%s.rel_extent"          },
    { LLM_KV_INKLING_REL_EXTENT_SWA,      "%s.rel_extent_swa"      },
    { LLM_KV_INKLING_SHORTCONV_KERNEL,    "%s.shortconv_kernel"    },
    { LLM_KV_INKLING_DENSE_BLOCK_COUNT,   "%s.dense_block_count"   },
    { LLM_KV_INKLING_LOGIT_SCALE_DENOM,   "%s.logit_scale_denom"   },
    { LLM_KV_INKLING_LOG_SCALING_N_FLOOR, "%s.log_scaling_n_floor" },
    { LLM_KV_INKLING_LOG_SCALING_ALPHA,   "%s.log_scaling_alpha"   },
    { LLM_KV_INKLING_UNPADDED_VOCAB_SIZE, "%s.unpadded_vocab_size" },
    // In-Place TTT; same spellings as gguf-py Keys.TTT (row S69-FORK-INPLACE-TTT-2026-10-26-01)
    { LLM_KV_TTT_LAYERS,                  "%s.ttt.layers"          },
    { LLM_KV_TTT_CHUNK,                   "%s.ttt.chunk"           },
    { LLM_KV_TTT_LR,                      "%s.ttt.lr"              },
    { LLM_KV_TTT_TARGET,                  "%s.ttt.target"          },
    { LLM_KV_TTT_CONV_PADDING,            "%s.ttt.conv_padding"    },
    { LLM_KV_TTT_CLIP_TAU,                "%s.ttt.clip_tau"        },
    { LLM_KV_TOKENIZER_NORMALIZER_LOWERCASE,     "tokenizer.ggml.normalizer.lowercase"     },
    { LLM_KV_TOKENIZER_NORMALIZER_STRIP_ACCENTS, "tokenizer.ggml.normalizer.strip_accents" },
    { LLM_KV_TOKENIZER_NORMALIZER_NFKC,          "tokenizer.ggml.normalizer.nfkc"          },
    { LLM_KV_TOKENIZER_DNA_AUTO_TAGS,            "tokenizer.ggml.dna_auto_tags"            },
    { LLM_KV_TOKENIZER_CLEAN_SPACES,             "tokenizer.ggml.clean_spaces"             },
    { LLM_KV_TOKENIZER_NORMALIZER_NFC,           "tokenizer.ggml.normalizer.nfc"           },
    { LLM_KV_TOKENIZER_METASPACE_PREPEND_SCHEME, "tokenizer.ggml.metaspace.prepend_scheme" },
    { LLM_KV_TOKENIZER_SUPPRESS_TOKENS,          "tokenizer.ggml.suppress_tokens"          },
    { LLM_KV_ADAPTER_LORA_TASK_NAME,          "adapter.lora.task_name"     },
    { LLM_KV_ADAPTER_LORA_PROMPT_PREFIX,      "adapter.lora.prompt_prefix" },
    { LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS, "adapter.alora.invocation_tokens" },
    { LLM_KV_ADAPTER_KINDS,                   "adapter.kinds"                   },
    { LLM_KV_ADAPTER_LORA_GATE,               "adapter.lora.gate"               },
    { LLM_KV_ADAPTER_LORA_GATE_TOKEN,         "adapter.lora.gate_token"         },

    { LLM_KV_XIELU_ALPHA_N,         "xielu.alpha_n"         },
    { LLM_KV_XIELU_ALPHA_P,         "xielu.alpha_p"         },
    { LLM_KV_XIELU_BETA,            "xielu.beta"            },
    { LLM_KV_XIELU_EPS,             "xielu.eps"             },

    { LLM_KV_PRISM_HADAMARD_VERSION,              "prism.hadamard.version"              },
    { LLM_KV_PRISM_HADAMARD_TIED_OUTPUT,          "prism.hadamard.tied_output"          },
    { LLM_KV_PRISM_HADAMARD_BLOCK_SIZE,           "prism.hadamard.block_size"           },
    { LLM_KV_PRISM_HADAMARD_TRANSFORM,            "prism.hadamard.transform"            },
    { LLM_KV_PRISM_HADAMARD_AXIS,                 "prism.hadamard.axis"                 },
    { LLM_KV_PRISM_HADAMARD_SIGN_MODE,            "prism.hadamard.sign_mode"            },
    { LLM_KV_PRISM_HADAMARD_SIGN_WIDTHS,          "prism.hadamard.sign_widths"          },
    { LLM_KV_PRISM_HADAMARD_SIGN_VALUES,          "prism.hadamard.sign_values"          },
    { LLM_KV_PRISM_HADAMARD_WEIGHT_NAMES,         "prism.hadamard.weight_names"         },
    { LLM_KV_PRISM_HADAMARD_INVERSE_WEIGHT_NAMES, "prism.hadamard.inverse_weight_names" },
    { LLM_KV_PRISM_HADAMARD_GDN_V_GROUPED,        "prism.hadamard.gdn_v_grouped"        },

    // deprecated
};

LLM_KV::LLM_KV(llm_arch arch, const char* suffix) : arch(arch), suffix(suffix) {}

// The suffix is APPENDED as ".<suffix>", as upstream llama-arch.cpp does. It used to be passed as a
// second format argument, which a name with no second %s (e.g. "tokenizer.chat_template") silently
// dropped: llama_model_chat_template(model, "rerank") then returned the DEFAULT chat template, and
// the server's /rerank scored the raw Jinja text for every document (measured: n_tokens 986 for
// every pair, scores ~0.52-0.58 regardless of relevance).
std::string LLM_KV::operator()(llm_kv kv) const {
    std::string name = ::format(LLM_KV_NAMES.at(kv), prefix != nullptr ? prefix : LLM_ARCH_NAMES.at(arch));
    if (suffix != nullptr) {
        name += ".";
        name += suffix;
    }
    return name;
}

// RESTORED 2026-09-09, slice 21. Body verbatim from 279b4bfaf^2:src/llama-arch.cpp:1032-1045.
// It decides which architectures may use recurrent-state rollback (cparams.n_rs_seq); an arch not
// listed has n_rs_seq clamped to 0 by src/llama-context.cpp:106, which logs when it clamps.
//
// THE LIST IS UPSTREAM'S, UNWIDENED, AND THAT IS DELIBERATE. This fork carries architectures
// upstream does not, some of them recurrent or hybrid (QWEN3NEXT, KIMI_LINEAR, MINIMAX_01 among
// them). Whether any of those support rollback is a real question that needs evidence about the
// model, not a guess in a switch -- and guessing wrong here would enable a feature on an
// architecture that cannot honour it. Omitting them fails the safe way: the feature stays off.
// All eight enums below were confirmed present in this fork's llama-arch.h before this was written.
bool llm_arch_supports_rs_rollback(const llm_arch & arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_GLM5_NEXT:
            return true;
        default:
            return false;
    }
}

const char * llama_model_arch_name(llm_arch arch) {
    auto it = LLM_ARCH_NAMES.find(arch);
    if (it == LLM_ARCH_NAMES.end()) {
        return "unknown";
    }
    return it->second;
}

const char * llm_arch_name(llm_arch arch) {
    return llama_model_arch_name(arch);
}

std::vector<llm_arch> llm_arch_all() {
    std::vector<llm_arch> ret;
    ret.reserve(LLM_ARCH_NAMES.size());
    for (const auto & kv : LLM_ARCH_NAMES) {
        ret.push_back(kv.first);
    }
    return ret;
}

bool llm_arch_is_recurrent(const llm_arch & arch) {
    switch (arch) {
    case LLM_ARCH_MAMBA:
    case LLM_ARCH_MAMBA2:
    case LLM_ARCH_RWKV6:
    case LLM_ARCH_RWKV6QWEN2:
    case LLM_ARCH_RWKV7:
    case LLM_ARCH_ARWKV7:
    case LLM_ARCH_MMFREELM:   // MatMul-free LM: MLGRU recurrent state
        return true;
    default:
        return false;
    }
}

bool llm_arch_is_hybrid(const llm_arch & arch) {
    switch (arch) {
    case LLM_ARCH_QWEN3NEXT:
    case LLM_ARCH_QWEN35MOE:
    case LLM_ARCH_QWEN35:
    case LLM_ARCH_QWEN4EXP:
    case LLM_ARCH_DEEPSEEK4:
    case LLM_ARCH_DEEPSEEK41:
    case LLM_ARCH_OPENPANGU:
    case LLM_ARCH_BAILINGMOE3:
    case LLM_ARCH_KIMI_K3:      // S69: KDA (recurrent) layers interleaved with MLA (attention)
    case LLM_ARCH_LFM2:         // S69: short-convolution layers interleaved with attention
    case LLM_ARCH_GRANITE_HYBRID: // S69: mamba2 layers interleaved with attention
    case LLM_ARCH_JAMBA:
    case LLM_ARCH_FALCON_H1:
    case LLM_ARCH_PLAMO2:
    case LLM_ARCH_LFM2MOE:
    case LLM_ARCH_NEMOTRON_H:
    case LLM_ARCH_NEMOTRON_H_MOE:
    case LLM_ARCH_KIMI_LINEAR:
    case LLM_ARCH_MINIMAX_01:
    case LLM_ARCH_GLM5NEXT:
    case LLM_ARCH_INKLING:      // S69 P1: attention in EVERY layer plus four short-conv histories per layer (hybrid state row)
    case LLM_ARCH_GLM5_NEXT:    // upstream src/models/glm5-next.cpp (merged 2026-11-04)
        return true;
    default:
        return false;
    }
}

bool llm_arch_is_dflash_family(const llm_arch & arch) {
    return arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_DFLASH2 || arch == LLM_ARCH_DFLASH_DRAFT;
}

bool llm_arch_requires_all_graph_output_rows(const llm_arch & arch) {
    return arch == LLM_ARCH_DFLASH2;
}

bool llm_arch_is_diffusion(const llm_arch & arch) {
    switch (arch) {
    case LLM_ARCH_DREAM:
    case LLM_ARCH_LLADA:
    case LLM_ARCH_LLADA_MOE:
    case LLM_ARCH_RND1:
    case LLM_ARCH_DIFFUSION_GEMMA:
        return true;
    default:
        return false;
    }
}

// these models pick weights, routing or input meaning per ubatch based on token vs embd input
bool llm_arch_supports_mixed_batch(const llm_arch & arch) {
    switch (arch) {
        case LLM_ARCH_COGVLM:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_GRANITE_SWITCH:
        case LLM_ARCH_EAGLE3:
        case LLM_ARCH_DFLASH:
        case LLM_ARCH_GEMMA4_ASSISTANT:
            return false;
        default:
            return true;
    }
}

// BUILD-level capability for LLAMA_SPLIT_MODE_TENSOR (upstream's meta-device loader). This fork has not ported it
// (row S69-ENGREV-UPSTREAM-MERGE-SPLIT-MODE-TENSOR-META-DEVICE-2026-11-04-01), so the loader refuses the mode by name
// (llama.cpp llm_load_tensors) and test-llama-archs reports its "Meta" row SKIP instead of failing. ONE predicate for both,
// so the refusal and the test cannot disagree (2026-11-09: they did -- gate b1109h, 4 archs rc -1). Flip it to true in the
// same commit that ports the loader; llm_arch_supports_sm_tensor below stays the per-arch list.
bool llm_build_supports_sm_tensor() {
    return false;
}

bool llm_arch_supports_sm_tensor(const llm_arch & arch) {
    switch (arch) {
        case LLM_ARCH_GROK:
        case LLM_ARCH_MPT:
        case LLM_ARCH_PLAMO2:
        case LLM_ARCH_MINICPM3:
        case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
        case LLM_ARCH_JAMBA:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_T5:
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
        case LLM_ARCH_GRANITE_HYBRID:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM: // [TAG_MOE_AMAX_SCALE]
        case LLM_ARCH_FLINT:  // [TAG_MOE_AMAX_SCALE]
        case LLM_ARCH_LAGUNA: // [TAG_MOE_AMAX_SCALE]
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_QWEN3TTS:
            return false;
        default:
            return true;
    }
}
