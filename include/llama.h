//
// Copyright (C) 2023-2025 The llama.cpp authors
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#ifndef LLAMA_H
#define LLAMA_H

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"
#include "gguf.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LLAMA_DEFAULT_SEED 0xFFFFFFFF

#define LLAMA_TOKEN_NULL -1

#define LLAMA_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define LLAMA_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define LLAMA_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define LLAMA_SESSION_MAGIC   LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION 10

#define LLAMA_STATE_SEQ_MAGIC   LLAMA_FILE_MAGIC_GGSQ
#define LLAMA_STATE_SEQ_VERSION 4

#define LLAMA_SERVER_MAGIC 0x6c6d7376u // 'lmsv'
#define LLAMA_SERVER_VERSION 1

// A default argument for C++ callers only (C has none). Used where this fork's signature carries a trailing parameter upstream's does
// not, so upstream's spelling compiles unchanged: the default is the value upstream's shorter function passes on.
#ifdef __cplusplus
#define LLAMA_CXX_DEFAULT(v) = v
#else
#define LLAMA_CXX_DEFAULT(v)
#endif

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_vocab;
    struct llama_model;
    struct llama_context;
    struct llama_sampler;

    typedef struct llama_memory_i * llama_memory_t;

    typedef int32_t llama_pos;
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;

    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE   = 0, // For models without vocab
        LLAMA_VOCAB_TYPE_SPM    = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        LLAMA_VOCAB_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        LLAMA_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
        LLAMA_VOCAB_TYPE_TEST   = 7, // Dummy tokenizer for testing: rolling hash of fixed-size chunks -> tokens, tokens -> hex
        LLAMA_VOCAB_TYPE_PLAMO3 = 8, // PLaMo-3 tokenizer with pre-segmentation and dynamic programming
    };

    // pre-tokenization types
    //enum llama_vocab_pre_type {
    //    LLAMA_VOCAB_PRE_TYPE_DEFAULT        = 0,
    //    LLAMA_VOCAB_PRE_TYPE_LLAMA3         = 1,
    //    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM   = 2,
    //    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,
    //    LLAMA_VOCAB_PRE_TYPE_FALCON         = 4,
    //    LLAMA_VOCAB_PRE_TYPE_MPT            = 5,
    //    LLAMA_VOCAB_PRE_TYPE_STARCODER      = 6,
    //    LLAMA_VOCAB_PRE_TYPE_GPT2           = 7,
    //    LLAMA_VOCAB_PRE_TYPE_REFACT         = 8,
    //    LLAMA_VOCAB_PRE_TYPE_COMMAND_R      = 9,
    //    LLAMA_VOCAB_PRE_TYPE_STABLELM2      = 10,
    //    LLAMA_VOCAB_PRE_TYPE_QWEN2          = 11,
    //    LLAMA_VOCAB_PRE_TYPE_OLMO           = 12,
    //    LLAMA_VOCAB_PRE_TYPE_DBRX           = 13,
    //    LLAMA_VOCAB_PRE_TYPE_SMAUG          = 14,
    //    LLAMA_VOCAB_PRE_TYPE_PORO           = 15,
    //    LLAMA_VOCAB_PRE_TYPE_CHATGLM3       = 16,
    //    LLAMA_VOCAB_PRE_TYPE_CHATGLM4       = 17,
    //    LLAMA_VOCAB_PRE_TYPE_VIKING         = 18,
    //    LLAMA_VOCAB_PRE_TYPE_JAIS           = 19,
    //    LLAMA_VOCAB_PRE_TYPE_TEKKEN         = 20,
    //    LLAMA_VOCAB_PRE_TYPE_SMOLLM         = 21,
    //    LLAMA_VOCAB_PRE_TYPE_CODESHELL      = 22,
    //    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM  = 28, //llama.cpp lists this as 28
    //    LLAMA_VOCAB_PRE_TYPE_GPT4O          = 29,
    //    LLAMA_VOCAB_PRE_TYPE_SUPERBPE       = 30,
    //    LLAMA_VOCAB_PRE_TYPE_TRILLION       = 31,
    //    LLAMA_VOCAB_PRE_TYPE_BAILINGMOE     = 32,
    //    LLAMA_VOCAB_PRE_TYPE_LLAMA4         = 33,
    //    LLAMA_VOCAB_PRE_TYPE_FALCON_3       = 34,
    //    LLAMA_VOCAB_PRE_TYPE_FALCON_E       = 35,
    //    LLAMA_VOCAB_PRE_TYPE_SEED_CODER     = 36, //llama.cpp lists this as 35
    //    LLAMA_VOCAB_PRE_TYPE_HUNYUAN        = 37, //llama.cpp lists this as 36
    //    LLAMA_VOCAB_PRE_TYPE_KIMI_K2        = 38, //llama.cpp lists this as 37
    //};

    enum llama_rope_type {
        LLAMA_ROPE_TYPE_NONE   = -1,
        LLAMA_ROPE_TYPE_NORM   = 0,
        LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        LLAMA_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE,
        LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum llama_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
        LLAMA_TOKEN_TYPE_NORMAL       = 1,
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
        LLAMA_TOKEN_TYPE_CONTROL      = 3,
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
        LLAMA_TOKEN_TYPE_UNUSED       = 5,
        LLAMA_TOKEN_TYPE_BYTE         = 6,
    };

    enum llama_token_attr {
        LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
        LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
        LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,
        LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,
        LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum llama_ftype {
        LLAMA_FTYPE_ALL_F32              = 0,
        LLAMA_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        LLAMA_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, // except 1d tensors
        // 36/37 are MAINLINE's values for these two and this tree left them free, exactly as it
        // left GGML_TYPE 34/35 free. A gguf stores general.file_type, so a divergent number here
        // would make our ternary files unreadable by mainline and vice versa.
        LLAMA_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors (ternary 1.69 bpw)
        LLAMA_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors (ternary 2.06 bpw)
        LLAMA_FTYPE_MOSTLY_MXFP4         = 38, // except 1d tensors, 38 to be compatible with mainline
        // Unsloth IQ1 narrow-grid ftypes. 64-66 are Unsloth's own numbers (SOURCE-READ:
        // unslothai/llama.cpp iq1-narrow-grids-b10775 c86ed2699 include/llama.h, MIT) and free here, so
        // general.file_type in a GGUF written by their quantizer reads back correctly.
        LLAMA_FTYPE_MOSTLY_IQ1_XS        = 64, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_XXS       = 65, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_XXXS      = 66, // except 1d tensors
        // MX block-scaled FP6/FP8 (OCP MX style, E8M0 scale per 32). ftype numbers 70-73 mirror the
        // ggml type ids 70-73 and are free in this enum (64-66 IQ1 narrow, 41/42 Bonsai, 43/44 relocated).
        LLAMA_FTYPE_MOSTLY_MXFP6_E2M3    = 70, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_MXFP6_E3M2    = 71, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_MXFP8_E4M3    = 72, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_MXFP8_E5M2    = 73, // except 1d tensors
        // TurboQuant WHT-rotated weights; 74/75 mirror the ggml ids (their files say file_type 43/44, which are
        // MXFP4_MOE / Q2_0 here -- general.file_type is only a label, tensors are typed individually)
        LLAMA_FTYPE_MOSTLY_TQ3_1S        = 74, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_TQ4_1S        = 75, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q1_0_G128     = 41, // except 1d tensors. 41 is THIS FORK's value and is
                                               // kept because it is written into existing ggufs.
                                               // (The "38 to be compatible with mainline" note that
                                               // used to sit here was copied from the MXFP4 line
                                               // above and did not describe this enumerator at all.
                                               // Upstream's Q2_0, which used 41, is relocated to 44.)
        LLAMA_FTYPE_MOSTLY_Q2_0_G128     = 42,
        //
        LLAMA_FTYPE_MOSTLY_Q6_0          = 135, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_BN        = 136, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_BN        = 137, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_K         = 138, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_K         = 139, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_K         = 140, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ5_K         = 141, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ6_K         = 142, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_KS        = 145, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_KL        = 146, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_KS        = 147, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_KSS       = 148, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q8_KV         = 149, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ5_KS        = 150, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_KT        = 151, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_KT        = 152, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_KT        = 153, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_KS        = 154, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_KL        = 155, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_KT        = 156, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_PQ2_0         = 157, // except 1d tensors (Prism ternary, group-128)
        LLAMA_FTYPE_MOSTLY_PTQ1_0        = 158, // except 1d tensors (Prism ternary, base-3)

        LLAMA_FTYPE_MOSTLY_Q1_0_G128_R8  = 160, // except 1d tensors, 38 to be compatible with mainline
        LLAMA_FTYPE_MOSTLY_PQ2_0_R8      = 161, // except 1d tensors (Prism ternary, group-128)
        LLAMA_FTYPE_MOSTLY_PTQ1_0_R8     = 162, // except 1d tensors (Prism ternary, base-3)
        LLAMA_FTYPE_MOSTLY_TQK6          = 163, // except 1d tensors (trellis, group-128)
        LLAMA_FTYPE_MOSTLY_TQK7          = 164, // except 1d tensors (trellis, group-128)
        LLAMA_FTYPE_MOSTLY_TQ2_T         = 165, // except 1d tensors (trellis, group-128)
        LLAMA_FTYPE_MOSTLY_TQP5          = 166, // except 1d tensors (trellis, group-128)
        LLAMA_FTYPE_MOSTLY_TQP6          = 167, // except 1d tensors (trellis, group-128)
        LLAMA_FTYPE_MOSTLY_TQP7          = 168, // except 1d tensors (trellis, group-128)
        LLAMA_FTYPE_MOSTLY_NVFP4_KS      = 169, // except 1d tensors (ik 30903876b: 4-bit float blocks + fp32 row scale; 163-168 are ik's TQ* ftypes, absent here)
        LLAMA_FTYPE_MOSTLY_Q4_0_R8       = 202, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q8_0_R8       = 207, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_0_R4       = 208, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K_R4       = 210, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_R4       = 211, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_R4       = 214, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_R4       = 216, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_K_R4       = 218, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4    = 219, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XS_R4     = 220, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4    = 223, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_S_R4      = 224, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_NL_R4     = 225, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_S_R4      = 226, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_M_R4      = 229, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_XS_R8     = 230, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_M_R4      = 231, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_0_R4       = 335, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_BF16_R16      = 232, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_BN_R4     = 337, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_K_R4      = 338, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_K_R4      = 339, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_K_R4      = 340, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ5_K_R4      = 341, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_KS_R4     = 345, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_KS_R16    = 346, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_KS_R16    = 347, // except 1d tensors (16-row interleaved iq3ks-class codebook 3-bit)
        LLAMA_FTYPE_MOSTLY_IQ5_KS_R16    = 348, // except 1d tensors (16-row interleaved IQ5_KS)
        LLAMA_FTYPE_MOSTLY_IQ5_KS_R4     = 350, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_MXFP4_R8      = 351, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_0_R8       = 352, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q8_K_R16      = 397, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q8_KV_R8      = 398, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q8_K_R8       = 399, // except 1d tensors
        //
        // upstream ftypes, merged 2026-09-04.
        // NVFP4 and Q1_0 KEEP upstream's own numbers because 39 and 40 were free here, so files
        // written by mainline with those ftypes remain readable.
        // MXFP4_MOE and Q2_0 are RELOCATED: upstream puts them at 38 and 41, which this tree
        // already uses for MOSTLY_MXFP4 and MOSTLY_Q1_0_G128, and our existing quantised files
        // depend on those. A divergent number here would make our files unreadable by mainline
        // and vice versa -- the same reasoning the 36/37 comment above records.
        LLAMA_FTYPE_MOSTLY_NVFP4         = 39, // except 1d tensors (upstream number preserved)
        LLAMA_FTYPE_MOSTLY_Q1_0          = 40, // except 1d tensors (upstream number preserved)
        LLAMA_FTYPE_MOSTLY_MXFP4_MOE     = 43, // except 1d tensors (upstream 38, relocated)
        LLAMA_FTYPE_MOSTLY_Q2_0          = 44, // except 1d tensors (upstream 41, relocated)
        LLAMA_FTYPE_MOSTLY_F8_E4M3       = 402, // except 1d tensors (upstream 42 is OUR Q2_0_G128, relocated)

        LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    // Get the model file type (quantization) as a string, e.g. "Q8_0" or "Q4_K - Medium"
    LLAMA_API const char * llama_ftype_name(enum llama_ftype ftype);

    enum llama_rope_scaling_type {
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
        LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3,
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum llama_pooling_type {
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMA_POOLING_TYPE_NONE = 0,
        LLAMA_POOLING_TYPE_MEAN = 1,
        LLAMA_POOLING_TYPE_CLS  = 2,
        LLAMA_POOLING_TYPE_LAST = 3,
        LLAMA_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
        LLAMA_POOLING_TYPE_ATTENTION = 5, // NV-Embed-style param-free latent-attention pooling (query = masked mean, softmax over tokens)
    };

    enum llama_attention_type {
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum llama_flash_attn_type {
        LLAMA_FLASH_ATTN_TYPE_AUTO     = -1,
        LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
        LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1,
    };

    LLAMA_API const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type);

    enum llama_split_mode {
        LLAMA_SPLIT_MODE_NONE    = 0, // single GPU
        LLAMA_SPLIT_MODE_LAYER   = 1, // split layers and KV across GPUs
        // ---- this fork's tensor-parallel modes. KEPT AT 2 AND 3 ----
        // Measured 2026-09-04: upstream's llm_arch_supports_sm_tensor is a DENY-list containing
        // DEEPSEEK2, BAILINGMOE3, MINIMAX_M2 and KIMI_K3, so LLAMA_SPLIT_MODE_TENSOR returns false
        // for those architectures. This fork's ATTN/GRAPH modes cover them -- tp_mode graph
        // builders exist for exactly those. Complementary coverage, so both are kept.
        LLAMA_SPLIT_MODE_ATTN    = 2, // splits self-attention computations across GPUs
        LLAMA_SPLIT_MODE_GRAPH   = 3, // splits computations across GPUs
        // ---- upstream's, relocated from 2 and 3 ----
        // ROW is deprecated in upstream's docs but common/arg.cpp still accepts `row` on the CLI,
        // so it is carried rather than dropped.
        LLAMA_SPLIT_MODE_ROW     = 4, // split weights across GPUs by rows (upstream 2, relocated)
        LLAMA_SPLIT_MODE_TENSOR  = 5, // split weights and KV across GPUs (upstream 3, relocated)
    };

    enum llama_load_mode {
        LLAMA_LOAD_MODE_AUTO       = -1, // auto-detect based on device capabilities
        LLAMA_LOAD_MODE_NONE       =  0, // no special loading mode
        LLAMA_LOAD_MODE_MMAP       =  1, // memory map the model
        LLAMA_LOAD_MODE_MLOCK      =  2, // force system to keep model in RAM
        LLAMA_LOAD_MODE_MMAP_MLOCK =  3, // mmap + keep in RAM
        LLAMA_LOAD_MODE_DIRECT_IO  =  4, // use direct I/O if available
    };

    LLAMA_API const char * llama_load_mode_name(enum llama_load_mode load_mode);
    LLAMA_API enum llama_load_mode llama_load_mode_from_str(const char * str);

    enum llama_lazy_mode {
        LLAMA_LAZY_MODE_OFF  = 0, // always read the whole tensor up front
        LLAMA_LAZY_MODE_AUTO = 1, // lazy only for marked tensors larger than 4 GiB (requires mmap)
        LLAMA_LAZY_MODE_ON   = 2, // read the rows of tensors marked by the arch on demand (requires mmap)
    };

    enum llama_context_type {
        LLAMA_CONTEXT_TYPE_DEFAULT = 0,
        LLAMA_CONTEXT_TYPE_MTP     = 1,
    };

    enum llama_mtp_op_type {
        MTP_OP_NONE             = 0,
        MTP_OP_WARMUP           = 1,
        MTP_OP_UPDATE_ACCEPTED  = 2,
        MTP_OP_DRAFT_GEN        = 3,
    };

    enum llama_ckpt_type {
        LLAMA_CHECKPOINT_SPEC = 0,
        LLAMA_CHECKPOINT_REWIND = 1,
    };

    typedef struct llama_ckpt_data {
        enum llama_ckpt_type type;
        llama_pos pos;
    } llama_ckpt_data;

    typedef struct llama_token_data {
        llama_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llama_token_data;

    typedef struct llama_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        llama_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;      // note: do not assume the data is sorted - always check this flag
    } llama_token_data_array;

    typedef bool (*llama_progress_callback)(float progress, void * user_data);

    // Input data for llama_encode/llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by llama_encode/llama_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits; // TODO: rename this to "output"
    } llama_batch;

    enum llama_model_kv_override_type {
        LLAMA_KV_OVERRIDE_TYPE_INT,
        LLAMA_KV_OVERRIDE_TYPE_FLOAT,
        LLAMA_KV_OVERRIDE_TYPE_BOOL,
        LLAMA_KV_OVERRIDE_TYPE_STR,
    };

    enum llama_model_meta_key {
        LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_K,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_P,
        LLAMA_MODEL_META_KEY_SAMPLING_MIN_P,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
        LLAMA_MODEL_META_KEY_SAMPLING_TEMP,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
    };

    enum llama_process_type {
        LLAMA_PROCESS_TYPE_ENCODE,
        LLAMA_PROCESS_TYPE_DECODE,
    };

    struct llama_model_kv_override {
        enum llama_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct llama_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    // NVFP4_KS activation precision (adapted from ik_llama.cpp b71fe5368, MIT).
    // AUTO: use the checkpoint's .input_scale when valid; W4A4: always dynamic (-1); INT8: never W4A4 (0).
    enum llama_nvfp4_precision {
        LLAMA_NVFP4_PRECISION_AUTO = 0,
        LLAMA_NVFP4_PRECISION_W4A4 = 1,
        LLAMA_NVFP4_PRECISION_INT8 = 2,
    };

    struct llama_model_params {
        // RENAMED from `devices` in this merge. Upstream's `devices` is a different TYPE
        // (ggml_backend_dev_t *), and a struct has one member per name, so the two cannot share
        // it. The array form kept the plain name because it has the wider live use; this string
        // form is kept because it expresses something the array cannot -- the fork's RPC device
        // addressing "server|device", built by add_rpc_devices() at common/common.cpp:655.
        // DONE 2026-09-13 (was "FOLLOW-UP OWED" here): common/common.cpp now writes
        // `mparams.device_spec = params.devices.c_str();` and src/llama.cpp parses it
        // (string_split on ','), so the fork's "server|device" RPC addressing reaches the loader.
        const char* device_spec;

        int32_t n_gpu_layers; // number of layers to store in VRAM
        int32_t mla;          // MLA implementation to use (only applicable to DeepSeek models at this point)
        enum llama_split_mode split_mode; // how to split the model across multiple GPUs

        // main_gpu interpretation depends on split_mode:
        // LLAMA_SPLIT_NONE: the GPU that is used for the entire model
        // LLAMA_SPLIT_ROW: the GPU that is used for small tensors and intermediate results
        // LLAMA_SPLIT_LAYER: ignored
        int32_t main_gpu;
        int32_t max_gpu;
        int32_t ncmoe;

        enum ggml_type type_k;
        enum ggml_type type_v;
        enum ggml_type idx_type_k;
        uint32_t max_ctx_size;
        int32_t  n_seq_max;
        int32_t  n_ubatch;
        int32_t  amb;
        int32_t  fit_margin;
        bool     fit;
        int32_t  worst_graph_tokens;
        enum ggml_type type_k_first;
        enum ggml_type type_k_last;
        enum ggml_type type_v_first;
        enum ggml_type type_v_last;
        int32_t n_k_first;
        int32_t n_k_last;
        int32_t n_v_first;
        int32_t n_v_last;

        enum ggml_type extra_output_type;

        // --- adopted from upstream in this merge ---
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices
        // are used). This keeps the plain name `devices` because every measured assignment to a
        // llama_model_params in this tree uses the array form; see the header comment of
        // temp/apply_llamah_block1_model_params.py for the 22 call sites.
        ggml_backend_dev_t * devices;

        // how to load the model
        enum llama_load_mode  load_mode;
        enum llama_lazy_mode  lazy_mode;  // upstream: on-demand reading of tensors marked by the arch (read at src/llama.cpp)

        // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        const float * tensor_split;

        const int   * fit_margin_array;

        // comma separated list of RPC servers to use for offloading
        const char * rpc_servers;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        llama_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct llama_model_kv_override * kv_overrides;

        const struct llama_model_tensor_buft_override * tensor_buft_overrides;

        // Already loaded model to take the shared token embeddings, output norm and lm head from,
        // for a draft head that ships none of them (a NextN/MTP sidecar exported with
        // `--mtp-shared-embd`). Only consulted when the file being loaded has no
        // `token_embd.weight`; a file that carries its own copy is never touched by this.
        // Must outlive the model being loaded. SOURCE-READ: unslothai/llama.cpp f36eda84
        // (include/llama.h `model_shared`), MIT.
        const struct llama_model * model_shared;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;    // only load the vocabulary, no weights
        bool use_mmap;      // use mmap if possible
        bool use_mlock;     // force system to keep model in RAM
        bool check_tensors; // validate model tensor data
        bool repack_tensors;// repack if available
        bool use_thp;       // use transparent huge pages (linux only)
        bool validate_quants; // if true, check for NaNs while loading the model
        bool merge_qkv;     // if true, merge separate Q, K, V tensors into a single, contiguous tensor
        bool merge_up_gate_exps;  // if true, merge ffn_up_exps and ffn_gate_exps tensors into a single, contiguous tensor
        bool mtp;           // if true, load MTP layers if present
        bool dry_run;       // skip loading tensors
        bool flash_attn;
        bool defer_experts;    // defer expert mmap residency to speed up model loading (Linux only)
        bool defer_ple;        // keep sparse tables (PLE, engram) on the file instead of resident in memory (Linux and Windows only)
        bool defer_ple_auto;   // upstream --lazy-mode auto: defer those tables only when they exceed 4 GiB (defer_ple forces it)
        bool swa_compress;     // must match llama_context_params::swa_compress; the fit also assumes that context's n_ubatch

        // --- adopted from upstream in this merge ---
        // vocab_only and check_tensors are NOT repeated here: ours declares both above, identically.
        bool use_extra_bufts; // use extra buffer types (used for weight repacking)
        bool no_host;         // bypass host buffer, allowing extra buffers to be used
        bool no_alloc;        // only load metadata and simulate memory allocations
        // PRODUCERLESS AS OF 2026-09-09, and left in place deliberately. Nothing in src/ reads
        // this field: the only two mentions are the default initialisers that set it to false.
        // The live switch is `mtp` above -- src/llama.cpp:5114 does `model.mtp = params.mtp`, and
        // src/llama-load-tensors.cpp:1875/:1986 branch on model.mtp. This one is upstream's
        // spelling, adopted by the merge without its consumers.
        // It is kept, not removed, because it is a public field in a public header and taking it
        // out is a separate decision with its own blast radius. Documented here instead so the
        // next reader does not wire new code to the inert one -- which is exactly how
        // hparams.n_layer_nextn accumulated 35 readers and no writer.
        bool load_mtp;        // whether to load MTP layers. NOT READ BY ANYTHING -- use `mtp`.
                              // REDUNDANT WITH OURS, and said plainly rather than glossed: this
                              // struct already declares `bool mtp; // if true, load MTP layers if
                              // present`, which means the SAME thing. (The `mtp` in
                              // llama_context_params is a different field -- "Activate MTP if
                              // supported" -- so there are two distinct concepts and three fields.)
                              // Both are kept because upstream's loader writes load_mtp and the
                              // fork's writes mtp, and dropping either breaks its own caller. They
                              // must therefore be kept in agreement, or one delegated to the other,
                              // when the loader TUs are resolved. That is real follow-up work and
                              // it is recorded in the row, not left to be discovered.

        // K smoothing (QServe "SmoothAttention", arXiv 2405.04532 section IV-B, with scales derived from the
        // WEIGHTS instead of calibration). 0 = off. > 0 = the migration exponent alpha (0.5 is the measured
        // choice). Per layer, K channel c is divided by s_c and Q channel c multiplied by s_c before RoPE, so q.k
        // is unchanged in exact arithmetic while the K CACHE holds a flatter distribution that quantizes better.
        // s is equal within each RoPE pair, so it commutes with the rotation.
        // Row S69-FORK-K-SMOOTHING-CALIBRATION-FREE-SMOOTHATTENTION-2026-09-27-01.
        float k_smooth_alpha;

        // NVFP4_KS activation precision, see enum llama_nvfp4_precision (default AUTO = 0).
        int32_t nvfp4_precision;
    };

    // --- adopted from upstream in this merge ---
    // Required, not optional: llama_context_params::samplers (adopted in the previous block) is a
    // pointer to this type, and MEASURED afterwards the active text had DEFINED=False /
    // MENTIONED=True for it, against a positive control that reported DEFINED=True.
    struct llama_sampler_seq_config {
        llama_seq_id           seq_id;
        struct llama_sampler * sampler;
    };

    // replacement policy of the MoE cache (llama_context_params::moe_cache_policy)
    //   SLRU:     segmented LRU, hits promote an expert to the protected half (the fork's original policy, the default)
    //   CLASSLRU: LRFU/MQ-like, one LRU list per frequency class (upstream llama.cpp #31233); the policy that can leave misses to the CPU
    enum llama_moe_cache_policy {
        LLAMA_MOE_CACHE_POLICY_SLRU     = 0,
        LLAMA_MOE_CACHE_POLICY_CLASSLRU = 1,
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       https://github.com/ggerganov/llama.cpp/pull/7544
    struct llama_context_params {
        uint32_t seed;              // RNG seed, -1 for random
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        int32_t n_threads;         // number of threads to use for generation
        int32_t n_threads_batch;   // number of threads to use for batch processing
        int32_t  max_extra_alloc;   // Max. additional VRAM the scheduler is allowed to allocate
        int32_t  worst_case_tokens; // number of tokens to use when reserving worst case graphs


        // --- adopted from upstream in this merge ---
        uint32_t n_rs_seq;              // recurrent-state snapshots per seq for rollback
                                        // (0 = no rollback) [EXPERIMENTAL]. NOT optional:
                                        // src/models/mamba-base.cpp reads cparams.n_rs_seq.
        uint32_t n_outputs_max;         // max outputs in a ubatch (0 = n_batch)
        uint32_t n_outputs_max_per_seq; // max outputs per sequence (0 = n_outputs_max)
        enum llama_context_type    ctx_type;        // context type (e.g. MTP)
        enum llama_flash_attn_type flash_attn_type; // when to enable Flash Attention
        enum llama_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
        enum llama_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum llama_attention_type    attention_type;    // attention type to use for embeddings

        // ref: https://github.com/ggerganov/llama.cpp/pull/2054
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]
        size_t moe_cache_size; // total device-cache budget for host-resident MoE experts, split across GPUs [EXPERIMENTAL]
        float  moe_cache_cpu_share; // share of the MoE cache misses computed on the CPU from the host experts instead of uploaded, 0 = all uploaded [EXPERIMENTAL]
        enum llama_moe_cache_policy moe_cache_policy; // replacement policy of the MoE cache, SLRU (default) or CLASSLRU [EXPERIMENTAL]
        enum ggml_type idx_type_k; // data type for indexer K cache [EXPERIMENTAL]
        enum ggml_type type_reduce; // data type for reduce operations
        enum ggml_type type_graph_attn; // flash-attn precision under -sm graph
        enum ggml_type type_k_first;
        enum ggml_type type_k_last;
        enum ggml_type type_v_first;
        enum ggml_type type_v_last;
        int32_t n_k_first;
        int32_t n_k_last;
        int32_t n_v_first;
        int32_t n_v_last;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool logits_all;  // the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
        bool flash_attn;  // whether to use flash attention [EXPERIMENTAL]
        int  mla_attn;    // whether to use MLA attention [EXPERIMENTAL]
        int  attn_max_batch;    // maximum batch size for attention computations [EXPERIMENTAL]
        bool fused_moe_up_gate; // whether to use fused MoE up/gate op
        bool grouped_expert_routing; // whether to use grouped expert routing (BailingMoeV2 arch)
        bool fused_up_gate;     // whether to use fused up/gate op [EXPERIMENTAL]
        bool fused_mmad;        // whether to use fused mul+multi_add op [EXPERIMENTAL]
        bool rope_cache;        // whether to use RoPE cache [EXPERIMENTAL]
        bool graph_reuse;       // whether to reuse graphs when possible [EXPERIMENTAL]
        bool dsa;               // enable GLM DSA sparse attention (off by default) [EXPERIMENTAL]
        bool fused_idx_topk;    // enable the fused indexer topk op (on by default) [EXPERIMENTAL]
        bool swa_compress;      // allocate sliding-window layers at window size instead of n_ctx (off by default) [EXPERIMENTAL]
        bool dsv4_legacy_state; // write DeepSeek-V4 state in the legacy full-slice layout (no MAGIC), byte-identical to main [EXPERIMENTAL]
        int  dsa_top_k;         // DSA top-k override (<0 => model's configured indexer_top_k) [EXPERIMENTAL]
        int  min_experts;
        float thresh_experts;
        bool only_active_experts;
        bool prefetch_experts;  // if true, stream mmap'd MoE expert weights into the page cache (Linux only)
        int  prefetch_experts_threads; // number of expert prefetch workers (<=0 = auto)
        bool k_cache_hadamard;  // if true, apply Hadamard transform to K-cache
        bool v_cache_hadamard;  // if true, apply Hadamard transform to V-cache (needs FA)
        int  kv_fresh_chunk;    // FA + quantized KV: 1 = attend the current ubatch from its unquantized K/V (fresh-chunk), 2 = A/B control
                                // (same dequantized-cache path, current ubatch read back from the cache); 0 = off
        bool split_mode_graph_scheduling; // if true, force split mode graph scheduling
        //bool split_mode_f16;    // if true, cast intermediate results to f16 before copying to other GPUs
        bool scheduler_async;   // if true, with split mode "graph" graph evaluation will be done using multiple threads
        bool mtp;   // Activate MTP if supported
        enum llama_mtp_op_type mtp_op_type;

        // Abort callback
        // if it returns true, execution of llama_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;
        void *              offload_policy;
        void *              cuda_params;

        // S69 #22/#66 — KV-eviction (SnapKV/H2O) + KVFlash paged-KV. 0 = off (default). Trailing fields so
        // existing ABI/field-order is unchanged; default_params aggregate-init zeros them.
        int32_t  snapkv_cap;          // SnapKV prompt-KV cap (0=off)
        int32_t  snapkv_window;       // SnapKV observation window
        int32_t  h2o_heavy;           // H2O heavy-hitter budget (0=off)
        int32_t  h2o_recent;          // H2O recent window kept
        int32_t  kvflash_pool_tokens; // KVFlash paged-KV resident pool tokens (0=off)
        bool     offload_attn;          // S69 #220 offload-attention (default false)
        int32_t  offload_attn_page;     // tokens per KV page
        float    offload_attn_budget;   // fraction of pages attended per step
        int32_t  offload_attn_selector; // 0 quest, 1 shadowkv, 2 retattn
        int32_t  offload_attn_sink;     // leading KV pages pinned regardless of score
        int32_t  offload_attn_recent;   // recent local window (pages ending at the query's page) kept in addition to the top-k; both graph paths
        int32_t  offload_attn_dense_layers; // the first N layers are never masked (attend densely); 0 = mask every layer
        int32_t  offload_attn_refresh;  // FA path: reuse the page selection for R single-token decode steps (1 = rescan every step)
        int32_t  offload_attn_scan;     // FA path page scan: 0 exact, 1 hierarchical bisect (approximate, fewer keys scored)
        int32_t  offload_attn_min_tokens; // floor on kept KV in TOKENS (0 = no floor)
        int32_t  offload_attn_head_reduce; // -1 = unset (env GGML_OFFLOAD_ATTN_HEAD_REDUCE or the built-in default), 0 union of per-head top-k, 1 max over heads
        char offload_attn_dense_spec[128]; // "" = use offload_attn_dense_layers (first N), a fixed array so draft contexts that copy the params own it; else "N" | "-N" (last N) | "0-2,45-47" (explicit set), see src/offload_attention_dense.h

        // ── TriAttention KV scoring params (unified ik_llama build) ─────────
        // Enable with --triattention <stats_path>. RESTORED 2026-08-20: the
        // implementation survived the consolidation and compiles into every
        // binary; these fields and the argv parsing did not, so the feature was
        // unreachable. Appended AFTER the existing trailing fields for the same
        // ABI reason the comment above gives -- field order up to this point is
        // unchanged.
        const char * tri_stats_path;  // pre-computed calibration stats (NULL = disabled)
        float        tri_budget;      // retention fraction (0.0-1.0), e.g. 0.75 = keep 75%
        int          tri_window;      // recent N tokens always kept (default 512)
        int          tri_interval;    // score/prune every N tokens (default 128)
        int          tri_sink;        // first N tokens kept as attention sinks (default 0)

        // --- adopted from upstream in this merge ---
        // embeddings and offload_kqv are deliberately NOT adopted here: llama_context_params
        // already declares both in the active text, and a second copy is a duplicate member.
        bool no_perf;     // measure performance timings
        bool op_offload;  // offload host tensor operations to device
        bool swa_full;    // use full-size SWA cache
        bool kv_unified;  // use a unified buffer across the input sequences when computing attention

        // [EXPERIMENTAL] backend sampler chain configuration
        // (the caller must keep the sampler chains alive; they must be chains, i.e.
        //  built with llama_sampler_chain_init)
        struct llama_sampler_seq_config * samplers;
        size_t                            n_samplers;

        // a source/target/parent context -- lets two contexts share results or llama_memory
        struct llama_context * ctx_other;

        // ── CATS activation sparsity (S69-CATS-STAGE1-2026-09-28) ───────────
        // Training-free magnitude threshold on the SwiGLU gate activation
        // (Lee et al., "CATS", arXiv:2404.08763, Sec. 3.1 eq. 4-5). Stage 1 is
        // CPU-only, quality-path (dense mask, no sparse matvec kernel).
        // Trailing fields, same ABI-preservation reason as tri_stats_path above.
        float        cats_sparsity;        // target sparsity in [0,1); 0 = off (default, byte-identical to baseline)
        const char * cats_thresholds_file; // path to calibration JSON (NULL = derive nothing; thresholds must be present when cats_sparsity>0)
        bool         cats_calibrate;       // internal: used by tools/cats-calibrate to force the un-fused
                                            // silu(gate) path without masking (see llama-cparams.h)
        // DISABLE-REVERSIBLE: moe_pool_slots (below) defaults to 0 = feature off; this is a
        // brand-new opt-in field block, not a capability being switched off.
        // ── MoE resident expert pool (S69-KT-EXPERT-POOL-2026-09-28) ─────────────
        // Appended AFTER ctx_other for the same ABI reason the SnapKV/TriAttention blocks
        // above give: existing field order/offsets up to ctx_other are unchanged.
        // Port of llama.cpp PR #27861 (csantiago78) + discussion #28248 (memoriaru); see
        // common/common.h for the full design note.
        int          moe_pool_slots;        // per-layer resident-expert slots (0 = disabled)
        int          moe_pool_inserts;      // max admissions applied per layer per step
        int          moe_pool_admit;        // recent-use admission threshold (1 = unconditional)
        bool         moe_pool_cpu_only;     // restrict pool to plain CPU/CPU_Mapped buffers
        const char * moe_stats_path;        // recorder output JSON (NULL = off)
        const char * moe_stats_mode;        // stat | stat_approx | per_pass | per_token
        const char * moe_placement_strategy;// uniform | frequency | front-loading | random
        const char * moe_placement_stats;   // recorded stats JSON for the frequency strategy
        bool         moe_dynamic_update;    // enable dynamic re-placement above the threshold
        int          moe_dynamic_threshold; // prefill token threshold for dynamic re-placement

        // FFD block-sparse decode (row S69-FFD-FASTER-FLASH-DECODING-SPARSE-DECODE-PICK-2026-09-28-01; BENCH-PENDING:
        // no speed is claimed, the name is the upstream repo's). Appended after the existing trailing fields for the
        // same ABI reason as above.
        bool         ffd;                   // keep a 2-bit per-block K shadow and select blocks on decode (default false)
        int32_t      ffd_block;             // cells per block (default 128)
        float        ffd_delta;             // keep a block whose estimated max score >= row max - delta (default 5.0)
        // KIVI simulated 2/4/8-bit KV quantisation on the f16 cache (src/llama-kv-kivi-sim.cpp). 0 = off (default, bit-identical).
        int32_t      kivi_bits;
        int32_t      kivi_group;            // tokens per K group / channels per V group (default 32)
        int32_t      kivi_residual;         // full-precision residual window in tokens (default 128, multiple of kivi_group)
        int32_t             dflash_query_capacity; // internal DFlash query capacity override

        // ── TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28) ───────────
        // Training-free INPUT magnitude threshold applied before a linear projection
        // (Liu et al., "TEAL", arXiv:2408.14690, Sec 4.2 Definition 1 / eq. for s_tp).
        // Stage 1 covers Wq/Wk/Wv (shared attn-norm input), Wgate/Wup (shared ffn-norm
        // input) and Wdown (gated-intermediate input) -- 6 of the paper's 7 matrices;
        // Wo is NOT wired this session (its input is only materialized inside the fused
        // llm_build_kv helper on most architectures) -- see findings doc, left dense.
        // Uniform per-tensor-family sparsity (no Algorithm 1 greedy block reallocation
        // this session) -- see findings doc for why.
        float        teal_sparsity;         // target sparsity in [0,1); 0 = off (default, byte-identical to baseline)
        const char * teal_thresholds_file;  // path to calibration JSON (tools/cats-calibrate --teal)
        bool         teal_calibrate;        // internal: forces the un-fused FFN path so ffn_mid can be observed

        // optional CPU affinity for the CPU worker threads (Linux only); thread t is
        // pinned to cpu_affinity[t % n_cpu_affinity], NULL = no pinning
        const int32_t * cpu_affinity;
        int32_t         n_cpu_affinity;

        // XAttention prefill block selection (row S69-openvino-PICK-XATTENTION-PREFILL-BLOCK-SELECT-2026-09-29-01). Appended last for the
        // same ABI reason as the fields above. Flash-attention path, single-sequence causal prefill only; 0 = off (default, no op spliced).
        float        xattn_threshold;       // fraction of each query block's estimated attention mass to keep, in (0,1); 0 = off
        int32_t      xattn_stride;          // anti-diagonal stride of the estimate (default 8; block must be a multiple)
        int32_t      xattn_block;           // selection block in tokens (default 128)

        // KVCrush / AdaptiveR-KV retention for the SnapKV/H2O prompt prune (row S69-openvino-PICK-KVCRUSH-DIVERSITY-RETENTION-2026-09-28-01).
        // Appended last (ABI). 0 = off: the prune keeps the mass-only plan.
        float        kv_diversity_mass;     // in (0,1]: fraction of the evictable attention mass kept by mass first; the rest of the budget
                                            // goes to the least redundant blocks (cosine similarity of the cached keys)
        int32_t      kv_diversity_block;    // selection block in tokens (default 16); the first block is always kept (attention sink)
        int32_t      kv_diversity_layers;   // evenly spaced layers whose keys are averaged into the redundancy matrix (default 4)

        // SnapKV/H2O survivor re-base (row S69-F1-PICK-SNAPKV-CONTIGUOUS-REBASE-AB-2026-10-06-01). Appended last (ABI).
        // 0 = keep the gapped original positions (default); 1 = shift the survivors to 0..keep-1 after the prune (K-shift on the next decode);
        // -1 = shift them the wrong way (+len), the negative control of examples/snapkv-rebase-ab.
        int32_t      snapkv_rebase;
    };

    // model quantization parameters
    struct quantize_user_data;

    // --- adopted from upstream in this merge ---
    // Declared BEFORE llama_model_quantize_params because that struct points at them.
    struct llama_model_tensor_override {
        const char * pattern;
        enum ggml_type type;
    };

    struct llama_model_imatrix_data {
        const char * name;
        const float * data;
        size_t size;
    };

    // ADOPTED AFTER A CORRECTION. These two were first marked "skip, already in the active text"
    // on the strength of a bare word search. That search was answering the wrong question: both
    // names are REFERENCED throughout llama.h -- as parameter and return types -- while their
    // typedefs live ONLY inside this conflict block. MEASURED by temp/_defined_vs_mentioned.py:
    // DEFINED=False, MENTIONED=True for both, against a positive control (llama_model_kv_override)
    // that reports DEFINED=True, so the detector is not simply saying no to everything. Skipping
    // them would have left the header referring to two types it never defines.
    typedef struct llama_logit_bias {
        llama_token token;
        float bias;
    } llama_logit_bias;

    typedef struct llama_sampler_chain_params {
        bool no_perf; // whether to measure performance timings
    } llama_sampler_chain_params;
    typedef struct llama_model_quantize_params {
        int32_t nthread;                     // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum llama_ftype ftype;              // quantize to this llama_ftype
        enum ggml_type output_tensor_type;   // output tensor type
        enum ggml_type token_embedding_type; // token embeddings tensor type
        enum ggml_type per_layer_token_embedding_type; // token embeddings tensor type
        enum ggml_type attn_q_type;          // attention query tensor type
        enum ggml_type attn_k_type;          // attention key tensor type
        enum ggml_type attn_v_type;          // attention value tensor type
        enum ggml_type attn_qkv_type;        // attention query-key-value tensor type
        enum ggml_type attn_output_type;     // attention output tensor type
        enum ggml_type ffn_gate_type;        // feedforward network gate type
        enum ggml_type ffn_down_type;        // feedforward network down type
        enum ggml_type ffn_up_type;          // feedforward network up type
        enum ggml_type ffn_gate_inp_type;    // routed experts probabilities type (relevant for MoE models only)
        enum ggml_type extra_output_type;    // routed experts probabilities type (relevant for MoE models only)
        bool allow_requantize;               // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;         // quantize output.weight
        bool only_copy;                      // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                           // quantize all tensors to the default type
        bool keep_split;                     // quantize to the same number of shards
        bool ignore_imatrix_rules;           // If set to true, the built-in rules for refusing to quantize into certain quants without imatrix are ignored
        bool only_repack;                    // Only repack tensors
        bool dry_run;                        //
        bool partial_requant;                // quantize only missing split files in the split quantized .gguf destination directory
        size_t slab_size;                    // tensors larger than this many bytes of f32 are processed in slabs of up to this size (at least one slice or row group), 0 = never
        bool cuda_quantize;                  // quantize IQ4_KT, IQ3_KT, IQ2_KT, IQ1_KT, IQ3_KS, IQ4_KS, IQ4_KS_R4, IQ4_KS_R16, IQ5_KS, IQ5_KS_R4 and IQ5_KS_R16 on a CUDA device
        int  cuda_quantize_device;           // CUDA device index used by --cuda-quantize (default 0)
        int  cuda_quantize_n_devices;        // row-split device count, 0 = off
        int  cuda_quantize_devices[16];      // row-split ordinals (-dev CUDA0,CUDA1,CUDA2)
        float cuda_quantize_split[16];       // row-split shares, sum 1 (-ts 3,3,2)
        void * imatrix_raw;                     // pointer to importance matrix data
        void * kv_overrides_raw;                 // pointer to vector containing overrides
        void * custom_quants;               // pointer to vector containing custom quantization rules
        void * repack_pattern;               // pointer to a vector containing regexes to be used for matching tensor names. Can be null
        struct quantize_user_data * user_data; // so we can pass extra data to the quantization functions

        // --- adopted from upstream in this merge ---
        // The fork's own void* forms of these two are kept above as imatrix_raw / kv_overrides_raw:
        // BOTH spellings have live consumers (two parallel quantize stacks), and a struct has one
        // member per name. The typed spellings keep the plain names so that kv_overrides means the
        // same thing here as it already does in llama_model_params.
        const struct llama_model_imatrix_data * imatrix;         // importance matrix data
        const struct llama_model_kv_override  * kv_overrides;    // kv overrides
        const struct llama_model_tensor_override * tt_overrides; // tensor type overrides
        const int32_t * prune_layers;                            // layer indices to prune
        size_t max_buf_size;                                     // upstream: max bytes of tensor rows kept in memory at once, 0 = default (8 GiB)

        // NVFP4 scale layout (S69-A-PICK-NVFP4-TWO-LEVEL-SCALE, 2026-10-05). Zero-initialised by the positional
        // default initialiser, so 0 = AUTO is the default. LLAMA_NVFP4_SCALE_ONE_LEVEL keeps the historic encoder (per-16
        // UE4M3 block scale only: block scales of real weights fall into the UE4M3 subnormal range, a smaller-magnitude
        // model flushes to zero); LLAMA_NVFP4_SCALE_TWO_LEVEL adds one FP32 per-tensor scale written as "<name>.scale";
        // AUTO picks TWO_LEVEL for architectures whose graph is known to apply that tensor and ONE_LEVEL otherwise.
        int32_t nvfp4_scale_mode;

        // Individual-shard quantize (rewrite of Thireus th/quantize_individual_tensors b07161d99/1ddb9c7ae, MIT;
        // the original changed llama_model_quantize's signature and made the option mandatory -- here it is two
        // optional fields, zero by default). Needs keep_split and a split input.
        //   only_splits / n_only_splits: 0-based split numbers. Only split 0 (the metadata split) and these input
        //     splits are opened -- the others need not exist on disk -- and only those output splits are written.
        //     Each written split is byte-identical to the same split of a full --keep-split run, so shards can be
        //     quantized one at a time or on different machines. Position-dependent k-quant mixtures (counters over
        //     attn_v / ffn_down) see only the loaded tensors, so use pure / custom quants for exact parity.
        //   skip_first_shard: do not write output split 0 (with only_splits, when split 0 is metadata only).
        const uint16_t * only_splits;
        int32_t          n_only_splits;
        bool             skip_first_shard;

        // GPTQ error-compensated rounding (llama-quantize --gptq-hessian; src/llama-gptq.h). Zero-initialised = off.
        //   gptq_hessian_raw: pointer to an imatrix_hessian_file (common/imatrix-hessian.h) loaded from llama-imatrix --hessian-block N
        //   gptq_damp: H_ii += damp * mean(diag H) before the factorisation (0 = 0.01, the ncnn / GPTQ default)
        //   gptq_mode: 0 off, 1 GPTQ, 2 the same block-wise calls with NO error propagation (the control arm)
        void *           gptq_hessian_raw;
        float            gptq_damp;
        int32_t          gptq_mode;
    } llama_model_quantize_params;

    enum llama_nvfp4_scale_mode {
        LLAMA_NVFP4_SCALE_AUTO      = 0,
        LLAMA_NVFP4_SCALE_ONE_LEVEL = 1,
        LLAMA_NVFP4_SCALE_TWO_LEVEL = 2,
    };

    // grammar types
    struct llama_grammar;



    // performance timing information
    struct llama_timings {
        double t_start_ms;
        double t_end_ms;
        double t_load_ms;
        double t_sample_ms;
        double t_p_eval_ms;
        double t_eval_ms;

        int32_t n_sample;
        int32_t n_p_eval;
        int32_t n_eval;
    };

    // used in chat template
    typedef struct llama_chat_message {
        const char * role;
        const char * content;
    } llama_chat_message;

    // lora adapter
    struct llama_lora_adapter;

    // Helpers for getting default parameters
    LLAMA_API struct llama_model_params llama_model_default_params(void);
    LLAMA_API struct llama_context_params llama_context_default_params(void);

    // --- adopted from upstream in this merge ---

    LLAMA_API const char * llama_version(void);
    LLAMA_API struct llama_sampler_chain_params  llama_sampler_chain_default_params(void);
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);

    // Initialize the llama + ggml backend
    // Call once at the start of the program
    LLAMA_API void llama_backend_init(void);

    //optional:
    LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);

    // Call once at the end of the program - currently only used for MPI
    LLAMA_API void llama_backend_free(void);

    LLAMA_API struct llama_model * llama_model_load_from_file(
                             const char * path_model,
            struct llama_model_params     params);

    LLAMA_API void llama_free_model(struct llama_model * model);

    // --- adopted from upstream in this merge ---

    // Optional: an auto threadpool gets created in ggml if not passed explicitly
    LLAMA_API void llama_attach_threadpool(
            struct llama_context * ctx,
               ggml_threadpool_t   threadpool,
               ggml_threadpool_t   threadpool_batch);

    LLAMA_API void llama_detach_threadpool(struct llama_context * ctx);

    // Create a new model from GGUF metadata as well as a function to set the tensor data
    //   - tensors are created as GGML_TYPE_F32 by default,
    //     override by adding a tensor with the same name but a different name to the context
    // RESTORED 2026-09-08. Upstream declares this typedef immediately before
    // llama_model_init_from_user (279b4bfaf^2:include/llama.h:488). The merge kept the FUNCTION
    // and dropped the TYPEDEF -- a plain omission, not a conflict: the fork had nothing here to
    // conflict with, since llama_model_init_from_user is upstream-new in its entirety. MSVC's
    // "syntax error: identifier 'llama_model_set_tensor_data_t'" is its generic fallback for a
    // token that should be a type name and is not.
    typedef void (*llama_model_set_tensor_data_t)(struct ggml_tensor * tensor, void * userdata);

    LLAMA_API struct llama_model * llama_model_init_from_user(
                    struct gguf_context * metadata,
          llama_model_set_tensor_data_t   set_tensor_data,    // function to initialize tensor data with
                                   void * set_tensor_data_ud, // userdata for function
              struct llama_model_params   params);

    DEPRECATED(LLAMA_API struct llama_model * llama_load_model_from_file(
                             const char * path_model,
              struct llama_model_params   params),
            "use llama_model_load_from_file instead");

    // Load a model from an open FILE pointer
    // The GGUF is read from the current position, so it can be embedded in a larger file
    // mmap needs the GGUF data section at a file offset to be aligned to the CPU tensor alignment (32 bytes)
    LLAMA_API struct llama_model * llama_model_load_from_file_ptr(
                                   FILE * file,
              struct llama_model_params   params);

    // Load a model from multiple splits (support custom naming scheme)
    // The paths must be in the correct order
    LLAMA_API struct llama_model * llama_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
              struct llama_model_params    params);

    LLAMA_API void llama_model_save_to_file(
            const struct llama_model * model,
                        const char * path_model);

    LLAMA_API void llama_model_free(struct llama_model * model);

    LLAMA_API struct llama_context * llama_init_from_model(
                     struct llama_model * model,
            struct llama_context_params   params);

    LLAMA_API void llama_set_offload_policy(struct llama_context * lctx, int op, bool on_or_off);
    DEPRECATED(LLAMA_API struct llama_context * llama_new_context_with_model(
                     struct llama_model * model,
            struct llama_context_params   params),
            "use llama_init_from_model instead");

    // Frees all allocated memory
    LLAMA_API void llama_free(struct llama_context * ctx);

    LLAMA_API int64_t llama_time_us(void);

    LLAMA_API size_t llama_max_devices(void);
    LLAMA_API size_t llama_max_parallel_sequences(void);
    LLAMA_API size_t llama_max_tensor_buft_overrides(void);

    LLAMA_API bool llama_supports_mmap       (void);
    LLAMA_API bool llama_supports_mlock      (void);
    LLAMA_API bool llama_supports_gpu_offload(void);



    LLAMA_API const struct llama_model * llama_get_model(const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ubatch   (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);

    LLAMA_API enum llama_pooling_type llama_pooling_type(const struct llama_context * ctx);

    struct llama_vocab;
    LLAMA_API enum llama_vocab_type   llama_vocab_type(const struct llama_vocab * vocab);
    LLAMA_API enum llama_rope_type    llama_rope_type   (const struct llama_model * model);

    LLAMA_API const struct llama_vocab* llama_model_get_vocab(const struct llama_model* model);
    LLAMA_API const char* llama_model_chat_template(const struct llama_model* model, const char* name);
    LLAMA_API int32_t llama_n_vocab    (const struct llama_model * model);
    LLAMA_API const struct llama_vocab* llama_get_model_vocab(const struct llama_model* model);
    LLAMA_API const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token);
    LLAMA_API int32_t llama_n_ctx_train(const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd     (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd_inp(const struct llama_model* model);

    LLAMA_API int32_t llama_n_layer    (const struct llama_model * model);

    // Compat
    LLAMA_API bool        llama_vocab_get_add_bos(const struct llama_vocab * vocab);
    LLAMA_API bool        llama_vocab_get_add_eos(const struct llama_vocab * vocab);
    LLAMA_API int32_t     llama_vocab_n_tokens(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab);

    // Get the model's RoPE frequency scaling factor
    LLAMA_API float llama_rope_freq_scale_train(const struct llama_model * model);

    // --- adopted from upstream in this merge (declarations only upstream had) ---
    LLAMA_API           llama_memory_t   llama_get_memory  (const struct llama_context * ctx);
    LLAMA_API const char * llama_model_cls_label(const struct llama_model * model, uint32_t i);
    LLAMA_API uint32_t llama_model_n_cls_out(const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_ctx_train  (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd_out   (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head       (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head_kv    (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_layer      (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_layer_nextn(const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_swa        (const struct llama_model * model);
    LLAMA_API float llama_model_rope_freq_scale_train(const struct llama_model * model);
    LLAMA_API enum llama_rope_type       llama_model_rope_type(const struct llama_model * model);
    LLAMA_API uint32_t llama_n_ctx_seq  (const struct llama_context * ctx);
    DEPRECATED(LLAMA_API int32_t llama_n_embd     (const struct llama_model * model), "use llama_model_n_embd instead");
    DEPRECATED(LLAMA_API int32_t llama_n_head     (const struct llama_model * model), "use llama_model_n_head instead");
    LLAMA_API uint32_t llama_n_rs_seq   (const struct llama_context * ctx);
    LLAMA_API bool llama_supports_rpc        (void);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLAMA_API int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLAMA_API int32_t llama_model_meta_count(const struct llama_model * model);

    // Get sampling metadata key name. Returns nullptr if the key is invalid
    LLAMA_API const char * llama_model_meta_key_str(enum llama_model_meta_key key);

    // Get metadata key name by index
    LLAMA_API int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLAMA_API int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    LLAMA_API uint64_t llama_model_size(const struct llama_model * model);

    // Per-device memory use, for the server's GET /memory and the memory_*_bytes gauges on /metrics.
    // Port of ik_llama.cpp 55813a5d7 (llama_memory_breakdown). Renamed llama_memory_breakdown_rows here because this
    // fork's src/llama-ext.h already owns the C++ type name llama_memory_breakdown (a buft -> data map).
    struct llama_memory_buft {
        ggml_backend_buffer_type_t buft;
        size_t                     size;
    };

    struct llama_memory_row {
        char   name[96];
        bool   is_device;
        size_t model;
        size_t context;
        size_t compute;
        size_t mmproj;
        size_t total;
        size_t free;
    };

    // Fills one row per offload device, then the host, then one row per other buffer type in use.
    // Sums the memory of all contexts in ctxs, such as a target and its draft contexts.
    // A context listed twice, or weights that several contexts share, are counted once.
    // Returns the number of rows, which can be larger than n_rows.
    LLAMA_API int32_t llama_memory_breakdown_rows(
            const struct llama_context * const * ctxs,
                                   int32_t   n_ctxs,
            const struct llama_memory_buft * mmproj,
                                   int32_t   n_mmproj,
                  struct llama_memory_row * rows,
                                   int32_t   n_rows);

    // Returns the total number of parameters in the model
    LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);

    // Get a llama model tensor
    LLAMA_API struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name);


    // --- adopted from upstream in this merge ---
    // Get the model file type (quantization), e.g. LLAMA_FTYPE_MOSTLY_Q8_0
    LLAMA_API enum llama_ftype llama_model_ftype(const struct llama_model * model);
    // Returns true if the model contains an encoder that requires llama_encode() call
    LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);

    // Returns true if the model contains a decoder that requires llama_decode() call
    LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    LLAMA_API bool llama_model_is_recurrent(const struct llama_model * model);

    // ---- DiffusionGemma (LLM_ARCH_DIFFUSION_GEMMA) block text-diffusion API -------------------
    // Every function below is a no-op / false / 0 on any other architecture.

    // true when the loaded model is a block text-diffusion model (canvas denoising, not AR decode)
    LLAMA_API bool llama_model_is_diffusion(const struct llama_model * model);

    // Per-request self-conditioning. `sc_logits` is the PREVIOUS step's raw canvas logits
    // [n_vocab * C]; it is BORROWED and must stay valid through the next llama_decode (that is when
    // the graph uploads it). use_sc is a {0,1} gate -- a block's FIRST denoising step must pass 0.
    // enabled=false is byte-identical to the zero-SC forward.
    LLAMA_API void llama_diffusion_set_sc(struct llama_model * model, const float * sc_logits,
                                          float use_sc, float temp_inv, bool enabled);

    // Opt into device-resident self-conditioning: keep the previous step's canvas logits on-device
    // instead of a per-step host upload. Same F32 values, so the forward stays bit-identical.
    LLAMA_API void llama_diffusion_set_device_sc(struct llama_model * model, bool enabled);

    // Select the prompt-KV-caching phase for the NEXT llama_decode.
    //   phase 0 = UNIFIED : no-cache forward over [prompt | canvas]  (default AND safety fallback)
    //   phase 1 = PREFILL : forward a prompt chunk starting at `off`, writing the K,V store
    //                       (`P` is the WHOLE prompt length and sizes the store)
    //   phase 2 = DECODE  : forward the canvas only, reading the cached prompt K,V (`P` = prompt len)
    LLAMA_API void llama_diffusion_set_phase(struct llama_model * model, int phase,
                                             int32_t P, int32_t off);

    // Bytes the prompt-KV store consumes per prompt token, so a caller can size context against it.
    // use_f16 must match the store type (flash-attn on -> F16).
    LLAMA_API size_t llama_diffusion_pkv_bytes_per_token(const struct llama_model * model, bool use_f16);

    // Stage-1 device sampling over the device-resident SC buffer: fills argmax/entropy/sampled for
    // n_tokens canvas positions without downloading the logits. Returns false when unavailable --
    // a build without CUDA, no device SC buffer allocated, or an architecture other than
    // DiffusionGemma -- so the caller uses the host path.
    LLAMA_API bool llama_diffusion_device_sample(const struct llama_model * model, const float * u,
                                                 int * argmax, float * entropy, int * sampled,
                                                 int n_tokens, float inv_temp);

    // Returns true if the model is hybrid (like Jamba, Granite, etc.)
    LLAMA_API bool llama_model_is_hybrid(const struct llama_model * model);

    LLAMA_API bool llama_model_has_recurrent(const struct llama_model * model);

    // Returns whether the model uses the DeepSeek-V4 architecture.
    LLAMA_API bool llama_model_is_deepseek4(const struct llama_model * model);

    // Returns true if the model is openPangu (conv-only recurrent state that rides the spec-rollback checkpoint)
    LLAMA_API bool llama_model_is_openpangu(const struct llama_model * model);

    LLAMA_API bool llama_kv_cache_is_compacted(const struct llama_context * ctx);

    LLAMA_API llama_pos llama_kv_cache_swa_rewind_floor(const struct llama_context * ctx);

    LLAMA_API llama_pos llama_kv_cache_n_swa(const struct llama_context * ctx);

    // Returns true if the model is a Gemma 4 MTP assistant (external frozen-KV speculative drafter)
    LLAMA_API bool llama_model_is_gemma4_mtp_assistant(const struct llama_model * model);

    LLAMA_API bool llama_model_is_step35(const struct llama_model * model);

    LLAMA_API bool llama_model_is_qwen35_family(const struct llama_model * model);

    LLAMA_API bool llama_model_is_qwen4exp(const struct llama_model * model);

    LLAMA_API bool llama_is_gemma4_mtp_file(const char * path);

    LLAMA_API bool llama_model_is_split_mode_graph(const struct llama_model * model);

    // Returns false for models whose KV cache cannot be re-positioned after the fact
    // (K-shift / context shift / self-extend), e.g. openPangu's latent cache.
    LLAMA_API bool llama_model_supports_ctx_shift(const struct llama_model * model);

    // Currently true for every model; no architecture is excluded from partial KV reuse.
    LLAMA_API bool llama_model_supports_partial_kv_reuse(const struct llama_model * model);

    // The complete answer for a context: the model-level query above, plus the context options it
    // cannot see (--swa-compress). Matches the gate the engine applies before a K-shift.
    LLAMA_API bool llama_supports_ctx_shift(const struct llama_context * ctx);

    LLAMA_API const char * llama_model_arch_string(const struct llama_model * model);

    // Returns 0 on success
    LLAMA_API uint32_t llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llama_model_quantize_params * params);

    // Load a LoRA adapter from file
    // The loaded adapter will be associated to the given model, and will be free when the model is deleted
    LLAMA_API struct llama_lora_adapter * llama_lora_adapter_init(
            struct llama_model * model,
            const char * path_lora);

    // Add a loaded LoRA adapter to given context
    // This will not modify model's weight
    LLAMA_API int32_t llama_lora_adapter_set(
            struct llama_context * ctx,
            struct llama_lora_adapter * adapter,
            float scale);

    // Remove a specific LoRA adapter from given context
    // Return -1 if the adapter is not present in the context
    LLAMA_API int32_t llama_lora_adapter_remove(
            struct llama_context * ctx,
            struct llama_lora_adapter * adapter);

    // Remove all LoRA adapters from given context
    LLAMA_API void llama_lora_adapter_clear(
            struct llama_context * ctx);

    // Manually free a LoRA adapter
    // Note: loaded adapters will be free when the associated model is deleted
    LLAMA_API void llama_lora_adapter_free(struct llama_lora_adapter * adapter);

    // --- adopted from upstream in this merge ---
    //
    // Adapters
    //

    // Load a LoRA adapter from file
    // The adapter is valid as long as the associated model is not freed
    LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init(
            struct llama_model * model,
            const char * path_lora);

    // Load a LoRA adapter from an open FILE pointer, reading from its current position
    LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init_from_file_ptr(
            struct llama_model * model,
            FILE * file);

    // Functions to access the adapter's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLAMA_API int32_t llama_adapter_meta_val_str(const struct llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLAMA_API int32_t llama_adapter_meta_count(const struct llama_adapter_lora * adapter);

    // Get metadata key name by index
    LLAMA_API int32_t llama_adapter_meta_key_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLAMA_API int32_t llama_adapter_meta_val_str_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Manually free a LoRA adapter
    // NOTE: loaded adapters that are not manually freed will be freed when the associated model is deleted
    LLAMA_API void llama_adapter_lora_free(struct llama_adapter_lora * adapter);

    // Get the invocation tokens if the current lora is an alora
    LLAMA_API uint64_t            llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter);
    LLAMA_API const llama_token * llama_adapter_get_alora_invocation_tokens  (const struct llama_adapter_lora * adapter);

    // true when the adapter is position-gated to draft rows (GGUF adapter.lora.gate = "draft_rows"): its delta reaches only the rows
    // of llama_set_draft_start, so the speculative draft pass uses it and the verify pass does not
    LLAMA_API bool llama_adapter_lora_is_draft_gated(const struct llama_adapter_lora * adapter);

    // The following functions operate on a llama_context, hence the naming: llama_verb_...

    // Per-SEQUENCE LoRa adapters: tokens of sequence `seq_id` get the listed adapters with the listed scales, ADDED to the
    // context-wide set of llama_set_adapters_lora (pass an empty context-wide set to use only these). One batch may mix
    // sequences that use different adapters: each token adds exactly the branches of its own sequence. n_adapters == 0
    // clears the sequence. Cost: every adapter that any sequence uses is evaluated on every token of the batch (its
    // branch is multiplied by a 0/scale mask), so this is meant for a few adapters at a time. Adapters must not be freed
    // while a sequence still lists them (clear the sequence first).
    LLAMA_API int32_t llama_set_seq_adapters_lora(
            struct llama_context * ctx,
            llama_seq_id seq_id,
            struct llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales);

    // Draft rows for position-gated LoRA (GGUF adapter.lora.gate = "draft_rows"): tokens of `seq_id` whose position is >= `pos`
    // are draft rows, and a gated adapter adds its delta to those rows only (every other row sees the base model). The state is
    // sticky and position-based, so it is the same however the batch is split into ubatches. pos < 0 clears the sequence.
    // Adapters with gate = "none" ignore it.
    LLAMA_API void llama_set_draft_start(
            struct llama_context * ctx,
            llama_seq_id seq_id,
            llama_pos pos);

    // Set LoRa adapters on the context. Will only modify if the adapters currently in context are different.
    LLAMA_API int32_t llama_set_adapters_lora(
            struct llama_context * ctx,
            struct llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales);

    // Apply a loaded control vector to a llama_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See llama_control_vector_load in common to load a control vector.
    LLAMA_API int32_t llama_set_adapter_cvec(
            struct llama_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    // The fork's original name for the same operation. KEPT, not replaced: it has a live caller at
    // examples/imatrix/imatrix.cpp:152 and its own definition at src/llama.cpp:9718, while
    // llama_set_adapter_cvec above has a live caller at common/common.cpp:4547 and a definition at
    // src/llama-context.cpp:3886. Two different names, so both can be declared and both callers
    // keep working. The parameter list below is shared and comes from the unconflicted lines.
    LLAMA_API int32_t llama_control_vector_apply(
            struct llama_context * lctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // KV cache
    //

    // Information associated with an individual cell in the KV cache view.
    struct llama_kv_cache_view_cell {
        // The position for this cell. Takes KV cache shifts into account.
        // May be negative if the cell is not populated.
        llama_pos pos;
    };

    // An updateable view of the KV cache.
    struct llama_kv_cache_view {
        // Number of KV cache cells. This will be the same as the context size.
        int32_t n_cells;

        // Maximum number of sequences that can exist in a cell. It's not an error
        // if there are more sequences in a cell than this value, however they will
        // not be visible in the view cells_sequences.
        int32_t n_seq_max;

        // Number of tokens in the cache. For example, if there are two populated
        // cells, the first with 1 sequence id in it and the second with 2 sequence
        // ids then you'll have 3 tokens.
        int32_t token_count;

        // Number of populated cache cells.
        int32_t used_cells;

        // Maximum contiguous empty slots in the cache.
        int32_t max_contiguous;

        // Index to the start of the max_contiguous slot range. Can be negative
        // when cache is full.
        int32_t max_contiguous_idx;

        // Information for an individual cell.
        struct llama_kv_cache_view_cell * cells;

        // The sequences for each cell. There will be n_seq_max items per cell.
        llama_seq_id * cells_sequences;
    };

    // work only with partial states, such as recurrent cache (e.g. Mamba)
#define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

    typedef uint32_t llama_state_seq_flags;

    // Create an empty KV cache view. (use only for debugging purposes)
    LLAMA_API struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_context * ctx, int32_t n_seq_max);

    // Free a KV cache view. (use only for debugging purposes)
    LLAMA_API void llama_kv_cache_view_free(struct llama_kv_cache_view * view);

    // Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
    LLAMA_API void llama_kv_cache_view_update(const struct llama_context * ctx, struct llama_kv_cache_view * view);

    // Returns the number of tokens in the KV cache (slow, use only for debug)
    // If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    LLAMA_API int32_t llama_get_kv_cache_token_count(const struct llama_context * ctx);

    // Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
    LLAMA_API int32_t llama_get_kv_cache_used_cells(const struct llama_context * ctx);

    // Clear the KV cache - both cell info is erased and KV data is zeroed
    LLAMA_API void llama_kv_cache_clear(
            struct llama_context * ctx);

    // Unified checkpoint API for recurrent/hybrid speculative decoding.
    enum llama_spec_ckpt_mode {
        LLAMA_SPEC_CKPT_NONE        = -1,
        LLAMA_SPEC_CKPT_AUTO        =  0,
        LLAMA_SPEC_CKPT_PER_STEP    =  1,
        LLAMA_SPEC_CKPT_GPU_FALLBACK =  2,
        LLAMA_SPEC_CKPT_CPU         =  3,
    };

    enum llama_spec_ckpt_restore_result {
        LLAMA_SPEC_CKPT_RESTORE_FAILED = 0,
        LLAMA_SPEC_CKPT_RESTORE_DIRECT = 1,
        LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED = 2,
    };

    // Initialise the checkpoint system for the upcoming speculation window.
    LLAMA_API int llama_spec_ckpt_init(struct llama_context * ctx, llama_ckpt_data ckpt_type, int mode, int max_tokens);

    // Save the current recurrent state as a speculative checkpoint.
    LLAMA_API bool llama_spec_ckpt_save(struct llama_context * ctx, llama_ckpt_data ckpt_type, llama_seq_id seq_id);

    // Save the current recurrent state as a rewind checkpoint
    LLAMA_API bool llama_rewind_ckpt_save(struct llama_context * ctx, llama_ckpt_data ckpt_type, llama_seq_id seq_id, int max_tokens);

    LLAMA_API bool llama_rewind_ckpt_discard(struct llama_context * ctx, llama_ckpt_data ckpt_type, llama_seq_id seq_id);

    LLAMA_API bool llama_rewind_ckpt_release(struct llama_context * ctx, llama_ckpt_data ckpt_type, llama_seq_id seq_id);


    // Restore the recurrent state after speculative decode.
    LLAMA_API bool llama_spec_ckpt_restore(struct llama_context * ctx, llama_ckpt_data ckpt_type, llama_seq_id seq_id,
                                            llama_pos n_past, int accepted_step);

    LLAMA_API enum llama_spec_ckpt_restore_result llama_spec_ckpt_restore_ex(
            struct llama_context * ctx, llama_ckpt_data ckpt_type, llama_seq_id seq_id,
            llama_pos n_past, int accepted_step);

    // Discard the saved checkpoint and reset internal mode state.
    LLAMA_API void llama_spec_ckpt_discard(struct llama_context * ctx, llama_ckpt_data ckpt_type);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    LLAMA_API bool llama_kv_cache_seq_rm(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_cache_seq_cp(
            struct llama_context * ctx,
                    llama_seq_id   seq_id_src,
                    llama_seq_id   seq_id_dst,
                       llama_pos   p0,
                       llama_pos   p1);

    // Removes all tokens that do not belong to the specified sequence
    LLAMA_API void llama_kv_cache_seq_keep(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // If the KV cache is RoPEd, the KV data is updated accordingly:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_cache_update()
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_cache_seq_add(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                       llama_pos   delta);

    // Integer division of the positions by factor of `d > 1`
    // If the KV cache is RoPEd, the KV data is updated accordingly:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_cache_update()
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_cache_seq_div(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                             int   d);

    // Returns the largest position present in the KV cache for the specified sequence
    LLAMA_API llama_pos llama_kv_cache_seq_pos_max(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Returns the smallest position present in the KV cache for the specified sequence
    LLAMA_API llama_pos llama_kv_cache_seq_pos_min(
        struct llama_context * ctx,
        llama_seq_id   seq_id);

    // Defragment the KV cache
    // This will be applied:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_cache_update()
    LLAMA_API void llama_kv_cache_defrag(struct llama_context * ctx);

    // Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
    // Positive return values does not mean a fatal error, but rather a warning.
    //    0 - success
    //    1 - Context overflow in a model where k-shift is not supported
    LLAMA_API int32_t llama_kv_cache_update(struct llama_context * ctx);

    // Memory
    //

    // Clear the memory contents
    // If data == true, the data buffers will also be cleared together with the metadata
    LLAMA_API void llama_memory_clear(
            llama_memory_t mem,
                      bool data);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence [TAG_LLAMA_SEQ_ID_NEG]
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    LLAMA_API bool llama_memory_seq_rm(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_cp(
            llama_memory_t mem,
              llama_seq_id seq_id_src,
              llama_seq_id seq_id_dst,
                 llama_pos p0,
                 llama_pos p1);

    // Removes all tokens that do not belong to the specified sequence
    LLAMA_API void llama_memory_seq_keep(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_add(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                 llama_pos delta);

    // Integer division of the positions by factor of `d > 1`
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_div(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                       int d);

    // Returns the smallest position present in the memory for the specified sequence
    // This is typically non-zero only for SWA caches
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LLAMA_API llama_pos llama_memory_seq_pos_min(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Returns the largest position present in the memory for the specified sequence
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LLAMA_API llama_pos llama_memory_seq_pos_max(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Check if the memory supports shifting
    LLAMA_API bool llama_memory_can_shift(llama_memory_t mem);

    // Get the memory object associated with this context
    LLAMA_API llama_memory_t llama_get_memory(const struct llama_context * ctx);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (logits, embedding and memory)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    LLAMA_API size_t llama_state_get_size(struct llama_context * ctx);
    LLAMA_API DEPRECATED(size_t llama_get_state_size(struct llama_context * ctx),
        "use llama_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    LLAMA_API size_t llama_state_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_copy_state_data(
            struct llama_context * ctx,
                         uint8_t * dst),
        "use llama_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    LLAMA_API size_t llama_state_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_set_state_data(
            struct llama_context * ctx,
                   const uint8_t * src),
        "use llama_state_set_data instead");

    // Save/load session file
    LLAMA_API bool llama_state_load_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    LLAMA_API DEPRECATED(bool llama_load_session_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use llama_state_load_file instead");

    LLAMA_API bool llama_state_save_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count);
    LLAMA_API DEPRECATED(bool llama_save_session_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count),
        "use llama_state_save_file instead");

    // Get the exact size needed to copy the KV cache of a single sequence
    LLAMA_API size_t llama_state_seq_get_size(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags LLAMA_CXX_DEFAULT(0));

    // Copy the KV cache of a single sequence into the specified buffer
    LLAMA_API size_t llama_state_seq_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags LLAMA_CXX_DEFAULT(0));

    // Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    LLAMA_API size_t llama_state_seq_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id,
           llama_state_seq_flags   flags LLAMA_CXX_DEFAULT(0));

    LLAMA_API size_t llama_state_seq_save_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   seq_id,
               const llama_token * tokens,
                          size_t   n_token_count);

    // Compare the model identity stored next to a slot file (sidecar "<filepath>.ident", written by llama_state_seq_save_file:
    // arch, n_layer, n_embd, n_head_kv, n_vocab, tensor count, and a fingerprint over the tensor directory + the first/last 4 KiB of
    // every tensor) with the model this context serves. Returns 0 = identical, 1 = MISMATCH (`why` names the first differing field;
    // llama_state_seq_load_file refuses such a file), 2 = no sidecar (file saved by an older build: cannot be proven wrong, restored with a warning).
    LLAMA_API int llama_state_seq_file_identity_check(
            struct llama_context * ctx,
                      const char * filepath,
                            char * why,
                          size_t   why_len);

    // If tokens_out is NULL, only the token count is reported through n_token_count_out and no state is loaded
    LLAMA_API size_t llama_state_seq_load_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   dest_seq_id,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

#define LLAMA_STATE_SEQ_FLAGS_NONE 0

// for backwards-compat
#define LLAMA_STATE_SEQ_FLAGS_SWA_ONLY 1

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

// Keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load).
// Getting the state for a seq_id with this flag invalidates all prior states gotten for that seq_id with this flag.
#define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE 2

    typedef uint32_t llama_state_seq_flags;

    LLAMA_API size_t llama_state_seq_get_size_ext(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_get_data_ext(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_set_data_ext(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id,
           llama_state_seq_flags   flags);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens
    // The sequence ID will be fixed to 0
    // The position of the tokens will be tracked automatically by llama_decode
    //
    // UPSTREAM'S ABI, adopted 2026-09-26 (DEC-S69-FORK-ADOPT-UPSTREAM-BATCH-ABI-2026-09-26): the fork's
    // older four-parameter form (pos_0, seq_id) and llama_batch's all_pos_0 / all_pos_1 / all_seq_id are
    // gone, as upstream removed them in ggerganov/llama.cpp#9745. For an explicit position or a sequence
    // other than 0, use llama_batch_init + common_batch_add instead -- NEVER drop the old arguments:
    // that is the silent seq_id bug upstream's reviewer caught in #9745 ("This will generate a batch for
    // seq_id == 0 and it needs to be seq_id == 1").
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    LLAMA_API struct llama_batch llama_batch_get_one(
                  llama_token * tokens,
                      int32_t   n_tokens);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with llama_batch_free()
    // If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    // The rest of the llama_batch members are allocated with size n_tokens
    // All members are left uninitialized
    LLAMA_API struct llama_batch llama_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with llama_batch_init()
    LLAMA_API void llama_batch_free(struct llama_batch batch);

    // Process a batch of tokens.
    // In contrast to llama_decode() - this call does not use KV cache.
    // For encode-decoder contexts, processes the batch using the encoder.
    // Can store the encoder output internally for later use by the decoder's cross-attention layers.
    //   0 - success
    // < 0 - error. the memory state is restored to the state before this call
    LLAMA_API int32_t llama_encode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Process a batch of tokens.
    // Requires the context to have a memory.
    // For encode-decoder contexts, processes the batch using the decoder.
    // Positive return values does not mean a fatal error, but rather a warning.
    // Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
    //   To handle this correctly, query the memory state using llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
    // Upon other return values, the memory state is restored to the state before this call
    //    0 - success
    //    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    //    2 - aborted     (processed ubatches will remain in the context's memory)
    //   -1 - invalid input batch
    // < -1 - fatal error (processed ubatches will remain in the context's memory)
    LLAMA_API int32_t llama_decode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    //
    // Extended batch API
    //

    struct llama_batch_ext;

    struct llama_embd {
        const float * data;
        size_t n_rows; // number of embedding rows in data
        size_t n_embd; // size of one row
    };

    LLAMA_API struct llama_batch_ext * llama_batch_ext_init (struct llama_context * ctx);
    LLAMA_API void                     llama_batch_ext_free (struct llama_batch_ext * batch);
    LLAMA_API void                     llama_batch_ext_clear(struct llama_batch_ext * batch);

    // Add an input token to the batch, with default values:
    //     id = LLAMA_TOKEN_NULL
    //     embd = nullptr
    //     pos = not set, the caller must set it with llama_batch_ext_set_pos()
    // Returns the batch index (>= 0)
    // On error:
    //     -1: batch is full
    //     -2: token is invalid (id == LLAMA_TOKEN_NULL or invalid embd)
    //     -3: invalid sequence id
    LLAMA_API int32_t llama_batch_ext_add      (struct llama_batch_ext * batch, llama_seq_id seq_id);

    // Add an input token to the batch, with a specified token ID or token embedding
    LLAMA_API int32_t llama_batch_ext_add_token(struct llama_batch_ext * batch, llama_seq_id seq_id, llama_token id);
    LLAMA_API int32_t llama_batch_ext_add_embd (struct llama_batch_ext * batch, llama_seq_id seq_id, struct llama_embd embd);

    // Add the token at index idx in the batch to another sequence id. The position will stays the same.
    // Note: this should be called before other _set() functions
    LLAMA_API bool llama_batch_ext_add_seq(
                                struct llama_batch_ext * batch,
                                               int32_t   idx,
                                          llama_seq_id   seq_id);

    // Set the token embedding for the token at index idx in the batch
    // use it after llama_batch_ext_add_token() to have an entry with both a token id and an embedding
    LLAMA_API bool llama_batch_ext_set_embd_token(
                                struct llama_batch_ext * batch,
                                               int32_t   idx,
                                     struct llama_embd   embd);

    // Set the "state" embedding for the token at index idx in the batch
    // "state" here means extra hidden state carried over from a previous stage, e.g.:
    //   - MTP: state from N layers of the target model
    //   - Qwen3 VL (deepstack): state from N layers of the vision encoder
    //   - LiliCorr: context embeddings for the transformers
    // Returns false if the context does not take a state embedding (MTP and LiliCorr contexts do)
    LLAMA_API bool llama_batch_ext_set_embd_state(
                                struct llama_batch_ext * batch,
                                               int32_t   idx,
                                     struct llama_embd   embd);

    // Set if output embeddings should be available for the token at index idx in the batch
    // Note: for now, this is equivalent to setting the output logits
    LLAMA_API bool llama_batch_ext_set_output_embd(
                                struct llama_batch_ext * batch,
                                               int32_t  idx,
                                                  bool  value);

    // Set output logits for the token at index idx in the batch
    // Note: for now, this is equivalent to setting the output embd
    LLAMA_API bool llama_batch_ext_set_output_logits(
                                struct llama_batch_ext * batch,
                                               int32_t  idx,
                                                  bool  value);

    // Set custom position for the token at index idx in the batch
    // For M-RoPE models:
    //     - Embedding tokens must have n_pos_per_embd positions per token, in order [t, y, x, z]; t is also the KV cache position
    //     - Text token only requires one single position per token
    LLAMA_API bool llama_batch_ext_set_pos(
                                struct llama_batch_ext * batch,
                                               int32_t   idx,
                                       const llama_pos * pos);

    // TODO: implement get_embeddings() and get_logits() for llama_batch_ext

    // Return values are the same as llama_decode()
    LLAMA_API int32_t llama_process(
                                struct llama_context * ctx,
                             enum llama_process_type   type,
                              struct llama_batch_ext * batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LLAMA_API void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    // pin CPU worker threads to the given logical CPUs (Linux only); thread t ->
    // cpus[t % n_cpus]. Configure before compute, n_cpus == 0 disables pinning.
    LLAMA_API void llama_set_cpu_affinity(struct llama_context * ctx, const int32_t * cpus, int n_cpus);

    // Get the number of threads used for generation of a single token.
    LLAMA_API int32_t llama_n_threads(struct llama_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    LLAMA_API int32_t llama_n_threads_batch(struct llama_context * ctx);

    // Set whether the context outputs embeddings or not
    // TODO: rename to avoid confusion with llama_get_embeddings()
    LLAMA_API void llama_set_embeddings(struct llama_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    LLAMA_API void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);

    // Returns whether the context is currently using causal attention
    LLAMA_API bool llama_get_causal_attn(const struct llama_context * ctx);

    // Set whether the model is in warmup mode or not
    // If true, all model tensors are activated during llama_decode() to load and cache their weights.
    //
    // note: using this can cause extra graph reallocations because it changes the graph topology with MoE models,
    //       so it is generally not recommended to use in practice. will be removed in the future
    DEPRECATED(LLAMA_API void llama_set_warmup(struct llama_context * ctx, bool warmup),
            "user code should do warmup runs manually [TAG_LLAMA_GRAPH_NO_WARMUP]");

    // Set abort callback
    LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    LLAMA_API void llama_synchronize(struct llama_context * ctx);

    // Token logits obtained from the last call to llama_decode()
    // The logits for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which llama_batch.logits[i] != 0
    // Cols: n_vocab
    // TODO: deprecate in favor of llama_get_logits_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLAMA_API float * llama_get_logits(struct llama_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indices can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

    // Restrict the output projection (lm_head) of the following llama_decode calls to a subset of the vocabulary: CTranslate2's vocabulary map,
    // Decoder::update_output_layer / Dense::select_weights (src/layers/decoder.cc:72-135, common.cc:313-335, MIT, SOURCE-READ). The rows of the
    // output weight listed in `ids` are copied once into a partial weight of the same type; every decode then multiplies by that partial weight
    // instead of the whole vocabulary. Logits of the listed tokens are what the full projection gives (same rows, same kernel); logits of all other
    // tokens are returned as -INFINITY. n_ids == 0, or a subset that covers the whole vocabulary, switches the restriction off. `ids` is sorted and
    // de-duplicated here. Returns false (and changes nothing) for an id outside the vocabulary or a model whose output weight cannot be row-gathered
    // (split across devices, row-interleaved quantisation, scale/rotation/LoRA attached to it, output bias).
    LLAMA_API bool llama_set_output_vocab_subset(struct llama_context * ctx, const llama_token * ids, int32_t n_ids);

    // Size of the vocabulary subset the LAST llama_decode really multiplied by (0 = the full projection was used, e.g. an architecture whose graph
    // does not go through the common output builder).
    LLAMA_API int32_t llama_output_vocab_subset_used(const struct llama_context * ctx);

    // Get the argmax token ID for DFlash draft position i without materializing full logits.
    // Returns LLAMA_TOKEN_NULL if argmax is not available (falls back to logits path).
    LLAMA_API llama_token llama_get_dflash_draft_token_ith(struct llama_context * ctx, int32_t i);

    // Materialize the DFlash draft logits on the host for llama_get_logits_ith(); off by default.
    LLAMA_API void llama_set_dflash_want_logits(struct llama_context * ctx, bool want);

    // Copy DFlash2 selector lattice after a draft decode with top_k*top_k scores
    // and top_k candidate IDs per position.
    LLAMA_API int32_t llama_get_dflash_draft_lattice_top_k(struct llama_context * ctx);
    LLAMA_API int32_t llama_get_dflash_draft_lattice_n_positions(struct llama_context * ctx);
    LLAMA_API bool llama_copy_dflash_draft_lattice(
            struct llama_context * ctx,
            float * scores, size_t score_count,
            int32_t * ids, size_t id_count);

    // Get all output token embeddings.
    // when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    // TODO: deprecate in favor of llama_get_embeddings_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indices can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    // shape: [n_embd] (1-dimensional)
    LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

    //
    // Vocab
    //

    LLAMA_API const char * llama_token_get_text(const struct llama_model * model, llama_token token);

    LLAMA_API float llama_token_get_score(const struct llama_model * model, llama_token token);

    LLAMA_API enum llama_token_attr llama_token_get_attr(const struct llama_model * model, llama_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    LLAMA_API bool llama_token_is_eog(const struct llama_model * model, llama_token token);
    LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);

    // Identify if Token Id is a control token or a render-able token
    LLAMA_API bool llama_token_is_control(const struct llama_model * model, llama_token token);

    // Special tokens
    LLAMA_API llama_token llama_token_bos(const struct llama_model * model); // beginning-of-sentence
    LLAMA_API llama_token llama_token_eos(const struct llama_model * model); // end-of-sentence
    LLAMA_API llama_token llama_token_cls(const struct llama_model * model); // classification
    LLAMA_API llama_token llama_token_sep(const struct llama_model * model); // sentence separator
    LLAMA_API llama_token llama_token_nl (const struct llama_model * model); // next-line
    LLAMA_API llama_token llama_token_pad(const struct llama_model * model); // padding

    // Returns -1 if unknown, 1 for true or 0 for false.
    LLAMA_API int32_t llama_add_bos_token(const struct llama_model * model);

    // Returns -1 if unknown, 1 for true or 0 for false.
    LLAMA_API int32_t llama_add_eos_token(const struct llama_model * model);

    // Codellama infill tokens
    LLAMA_API llama_token llama_token_prefix(const struct llama_model * model); // Beginning of infill prefix
    LLAMA_API llama_token llama_token_middle(const struct llama_model * model); // Beginning of infill middle
    LLAMA_API llama_token llama_token_suffix(const struct llama_model * model); // Beginning of infill suffix
    LLAMA_API llama_token llama_token_eot   (const struct llama_model * model); // End of infill middle

    // --- adopted from upstream in this merge ---

    //
    // backend sampling API [EXPERIMENTAL]
    // note: use only if the llama_context was created with at least one llama_sampler_seq_config
    //

    // Get the backend sampled token for the ith token.
    // With multiple outputs, sampler state advances when the token is accepted,
    // not when it is read through this function.
    // When accepting multiple outputs, accept a contiguous prefix in output order.
    // Returns LLAMA_TOKEN_NULL if no token was sampled.
    LLAMA_API llama_token llama_get_sampled_token_ith(struct llama_context * ctx, int32_t i);

    // Get the backend sampled probabilities for the ith token
    // The index matches llama_get_sampled_token_ith().
    // Returns NULL if no probabilities were generated.
    LLAMA_API float *  llama_get_sampled_probs_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t llama_get_sampled_probs_count_ith(struct llama_context * ctx, int32_t i);

    // Get the backend sampled logits for the ith token
    // Returns NULL if no logits were sampled.
    LLAMA_API float *  llama_get_sampled_logits_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t llama_get_sampled_logits_count_ith(struct llama_context * ctx, int32_t i);

    // Get the backend sampled candidates (token ids) for the ith token
    // These are needed to map probability/logit indices to vocab token ids.
    // Returns NULL if no candidates were sampled.
    LLAMA_API llama_token * llama_get_sampled_candidates_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t      llama_get_sampled_candidates_count_ith(struct llama_context * ctx, int32_t i);

    LLAMA_API float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token);

    // Identify if Token Id is a control token or a render-able token
    LLAMA_API bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token);
    LLAMA_API llama_token llama_vocab_eot(const struct llama_vocab * vocab); // end-of-turn
    LLAMA_API llama_token llama_vocab_sep(const struct llama_vocab * vocab); // sentence separator
    LLAMA_API llama_token llama_vocab_nl (const struct llama_vocab * vocab); // next-line
    LLAMA_API llama_token llama_vocab_pad(const struct llama_vocab * vocab); // padding
    LLAMA_API llama_token llama_vocab_mask(const struct llama_vocab * vocab); // mask
    LLAMA_API bool llama_vocab_get_add_sep(const struct llama_vocab * vocab);

    // model-specific suppress tokens (gguf key: tokenizer.ggml.suppress_tokens)
    LLAMA_API const llama_token * llama_vocab_get_suppress_tokens(const struct llama_vocab * vocab, int32_t * n_suppress_tokens);

    LLAMA_API llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab);
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pre(const struct llama_vocab * vocab), "use llama_vocab_fim_pre instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_suf(const struct llama_vocab * vocab), "use llama_vocab_fim_suf instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_mid(const struct llama_vocab * vocab), "use llama_vocab_fim_mid instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pad(const struct llama_vocab * vocab), "use llama_vocab_fim_pad instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_rep(const struct llama_vocab * vocab), "use llama_vocab_fim_rep instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_sep(const struct llama_vocab * vocab), "use llama_vocab_fim_sep instead");

    // CLS is equivalent to BOS
    DEPRECATED(LLAMA_API llama_token llama_vocab_cls(const struct llama_vocab * vocab), // classification
            "use llama_vocab_bos instead");

    //
    // Tokenization
    //
    // The API is thread-safe.
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    LLAMA_API int32_t llama_tokenize(
        const struct llama_model * model,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);
    LLAMA_API int32_t llama_vocab_tokenize(
        const struct llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    LLAMA_API int32_t llama_token_to_piece(
              const struct llama_model * model,
                           llama_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);
    LLAMA_API int32_t llama_token_to_piece_vocab(
              const struct llama_vocab * vocab,
                           llama_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of llama_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    LLAMA_API int32_t llama_detokenize(
        const struct llama_vocab * vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    ///
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat.
    /// @param chat Pointer to a list of multiple llama_chat_message
    /// @param n_msg Number of llama_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    LLAMA_API int32_t llama_chat_apply_template(
                            const char * tmpl,
       const struct llama_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);
    // Get list of built-in chat templates
    LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);

    typedef void* llama_sampler_context_t;

    struct llama_sampler;

    // The fork's OWN copies of struct llama_sampler_i and struct llama_sampler stood here and
    // were REMOVED 2026-09-08. The merge kept both sides' text, so this header defined each type
    // twice: error C2011 'llama_sampler_i': 'struct' type redefinition, and the same for
    // llama_sampler.
    //
    // NOT a two-sided divergence -- checked before deleting, because deleting the wrong copy
    // silently drops a capability. Upstream's definition (further down this file) carries all six
    // of the fork's methods with identical semantics PLUS six experimental backend-sampling
    // methods (backend_init/accept/apply/set_input/reset and copy_state). The fork's copy has
    // nothing upstream's lacks, so upstream's is a strict superset and is the one kept.
    //
    // Nothing between here and that definition needs the complete type: the forward declaration
    // `struct llama_sampler;` above covers every pointer use, and a scan of the intervening range
    // found five apparent references that are all COMMENTS matching the substring inside
    // llama_sampler_init_* / llama_sampler_chain_add. Real uses in that range: zero.

    //
    // Grammar
    //

    /// Initialize a llama_grammar.
    ///
    /// @param rules The rule elements of the grammar to initialize.
    /// @param n_rules The number of rules.
    /// @param start_rule_index The index of the root rule (the starting point of the grammar).
    /// @return The initialized llama_grammar or nullptr if initialization failed.
    //LLAMA_API struct llama_grammar * llama_grammar_init(
    //        const llama_grammar_element ** rules,
    //                             size_t    n_rules,
    //                             size_t    start_rule_index);

    struct llama_sampler_grammar;
    // REMOVED 2026-09-08: `LLAMA_API void llama_grammar_init_lazy(struct llama_sampler_grammar *);`
    //
    // PRIOR-ART-NONE: searched before removing, and named so the search is checkable --
    //   `git grep -w llama_grammar_init_lazy 279b4bfaf^2` (upstream ggml-org/llama.cpp at the
    //   merge base): no such symbol exists upstream, so there is no reference implementation to
    //   adopt. `git grep -w llama_grammar_init_lazy 279b4bfaf^1` (the fork parent): declared and
    //   undefined there too, so it is pre-merge, not merge fallout. `git grep -w` over the whole
    //   tree: the ONLY occurrence was this line -- zero callers (positive control:
    //   llama_grammar_free, three lines below, returns real callers from the same search).
    //   Read in full before deciding: llama_grammar_init_impl and the lazy path in
    //   src/llama-sampler.cpp:2660-2851, and struct llama_grammar at src/llama-grammar.h:126-151.
    //
    // The capability is NOT being switched off -- lazy grammars work and stay reachable. What is
    // removed is a promise nothing kept and nothing used. Every implementation this signature
    // admits is wrong, which is a stronger reason than "hard": the lazy machinery is
    // llama_grammar_init_impl(vocab, str, root, lazy, trigger_patterns, n, trigger_tokens, n), and
    // struct llama_grammar stores `lazy`, `awaiting_trigger`, `trigger_tokens` and
    // `trigger_patterns`, with `awaiting_trigger` documented there as "Initialized to true for
    // lazy grammars only". A function given ONLY the sampler pointer cannot supply triggers, so
    // all it could do is set lazy = true with an EMPTY trigger set -- a grammar that waits forever
    // for a trigger that can never match, silently dropping grammar constraint for the rest of
    // generation. A plausible-looking stub that disables a constraint is worse than a link error;
    // the same reasoning is recorded at src/llama-model.cpp:2642-2653 for llama_model_get_device.
    //
    // The working, trigger-taking entry points are unchanged and are what callers should use:
    //   llama_sampler_init_grammar_lazy(vocab, str, root, trigger_words, n, trigger_tokens, n)
    //   llama_sampler_init_grammar_lazy_patterns(vocab, str, root, patterns, n, tokens, n)
    // Row: S69-UPSTREAM-SRC-FILE-SPLIT-ARRIVED-UNWIRED-66-PUBLIC-SYMBOLS-2026-09-08-01.

    LLAMA_API void llama_grammar_free(struct llama_grammar * grammar);

    LLAMA_API struct llama_grammar * llama_grammar_copy(const struct llama_grammar * grammar);

    // A copy of the grammar's current state bound to another model's vocab, for a draft model whose vocab is
    // compatible with the target's (same token ids and pieces). A grammar checks every accepted token against the
    // vocab it was built with (src/llama-grammar.cpp llama_grammar_accept_impl), so a plain copy used through the draft
    // context aborts. Row S69-LMDEPLOY-PICK-GRAMMAR-MASKED-SPEC-DRAFT-2026-10-18-01.
    LLAMA_API struct llama_grammar * llama_grammar_copy_for_model(const struct llama_grammar * grammar,
                                                                  const struct llama_model * model);

    /// @details Apply constraints from grammar
    LLAMA_API void llama_grammar_apply(
            const struct llama_grammar * grammar,
            const struct llama_context * ctx,
                llama_token_data_array * candidates);
    LLAMA_API DEPRECATED(void llama_sample_grammar(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
      const struct llama_grammar * grammar),
        "use llama_grammar_apply instead");

    /// @details Accepts the sampled token into the grammar
    LLAMA_API void llama_grammar_accept_token(
            struct llama_grammar * grammar,
            struct llama_context * ctx,
                     llama_token   token);

    //
    // Sampling functions
    //

    // Sets the current rng seed.
    LLAMA_API void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed);

    /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
    LLAMA_API void llama_sample_repetition_penalties(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
               const llama_token * last_tokens,
                          size_t   penalty_last_n,
                           float   penalty_repeat,
                           float   penalty_freq,
                           float   penalty_present);

    /// @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
    /// @param logits Logits extracted from the original generation context.
    /// @param logits_guidance Logits extracted from a separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
    /// @param scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
    LLAMA_API void llama_sample_apply_guidance(
              struct llama_context * ctx,
                             float * logits,
                             float * logits_guidance,
                             float   scale);

    /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    LLAMA_API void llama_sample_softmax(
            struct llama_context * ctx,
          llama_token_data_array * candidates);

    LLAMA_API void llama_sample_dist(
        struct llama_context * ctx,
        llama_token_data_array * candidates);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API void llama_sample_top_k(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                         int32_t   k,
                          size_t   min_keep);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API void llama_sample_top_p(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   p,
                          size_t   min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
    LLAMA_API void llama_sample_min_p(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   p,
                          size_t   min_keep);

    /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
    LLAMA_API void llama_sample_tail_free(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   z,
                          size_t   min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_API void llama_sample_typical(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   p,
                          size_t   min_keep);

    /// @details Dynamic temperature implementation described in the paper https://arxiv.org/abs/2309.02772.
    LLAMA_API void llama_sample_entropy(
            struct llama_context * ctx,
          llama_token_data_array * candidates_p,
                           float   min_temp,
                           float   max_temp,
                           float   exponent_val);

    LLAMA_API void llama_sample_temp(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   temp);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    LLAMA_API void llama_sample_xtc(
            struct llama_context * ctx,
          llama_token_data_array * candidates_p,
                           float   probability,
                           float   threshold,
                           size_t  min_keep);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    LLAMA_API void llama_sample_top_n_sigma(
            struct llama_context * ctx,
          llama_token_data_array * candidates_p,
                           float   top_n_sigma);


LLAMA_API void                   llama_sampler_reset(struct llama_sampler* smpl);

/// @details Initializes a GBNF grammar, see grammars/README.md for details.
/// @param vocab The vocabulary that this grammar will be used with.
/// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
/// @param grammar_root The name of the start symbol for the grammar.

/// @details Lazy grammar sampler, introduced in https://github.com/ggerganov/llama.cpp/pull/9639
/// @param trigger_words A list of words that will trigger the grammar sampler. This may be updated to a loose regex syntax (w/ ^) in a near future.
/// @param trigger_tokens A list of tokens that will trigger the grammar sampler.


/// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
/// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
/// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982

    //LLAMA_API void llama_sample_dry(struct llama_context* ctx, llama_token_data_array* candidates_p, int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const char** seq_breakers, size_t num_breakers);

    void llama_sample_dry(struct llama_context* ctx, struct llama_sampler_dry* smpl, llama_token_data_array* candidates_p);

    // ADDED 2026-09-08, and it is a RENAME of an existing function, not a new capability.
    //
    // This is the constructor for the fork's standalone DRY sampler -- the family declared on the
    // next four lines, which common/sampling.cpp uses at :164, :196, :225, :262, :266 and :742.
    // It was called `llama_sampler_init_dry` and defined at src/llama.cpp:13651, which collided
    // head-on with upstream's sampler-chain function of the same name declared below at :2400 and
    // defined at src/llama-sampler.cpp:3539:
    //     src/llama.cpp:13635:28: error: ambiguating new declaration of
    //         'llama_sampler_dry* llama_sampler_init_dry(const llama_vocab*, float, float,
    //          int32_t, int32_t, const char**, size_t)'
    // Same name, same parameters, different return type. Both are live and both are wanted: the
    // chain form is what include/llama.h promises and what a llama_sampler chain needs, the
    // standalone form is what the fork's own sampling path is built on.
    //
    // The fork's one is renamed rather than the upstream one because upstream's name is fixed by
    // the public API and by decision #4 for this merge (adopt the sampler chain in full), and
    // because `llama_sampler_dry_init` is the spelling its own four siblings already use --
    // _reset, _free, _clone, _accept. It was the odd one out.
    struct llama_sampler_dry* llama_sampler_dry_init(const struct llama_vocab* vocab, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const char** seq_breakers, size_t num_breakers);

    void llama_sampler_dry_reset(struct llama_sampler_dry* smpl);

    void llama_sampler_dry_free(struct llama_sampler_dry* smpl);

    struct llama_sampler_dry* llama_sampler_dry_clone(struct llama_sampler_dry* smpl);

    void llama_sampler_dry_accept(struct llama_sampler_dry* smpl, llama_token token);

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982


    /// @details Adaptive p sampler initializer
    /// @param target Select tokens near this probability (valid range 0.0 to 1.0; <0 = disabled)
    /// @param decay Decay rate for target adaptation over time. lower values -> faster but less stable adaptation. (valid range 0.0 to 1.0; ≤0 = no adaptation)
    LLAMA_API struct llama_sampler_adaptive_p * llama_init_adaptive_p(
           const float target,
           const float decay,
            const bool updt_w_cur,
        const uint32_t seed);

    void llama_prep_adaptive_p(struct llama_context * ctx,
                 llama_token_data_array * candidates,
        struct llama_sampler_adaptive_p * adapt_p_ctx);

    /// @details Adaptive p sampler described in https://github.com/MrJackSpade/adaptive-p-docs/blob/main/README.md
    void llama_sample_adaptive_p(struct llama_context * ctx,
                               llama_token_data_array * candidates,
                      struct llama_sampler_adaptive_p * adapt_p_ctx);

    void llama_review_adaptive_p(struct llama_sampler_adaptive_p * adapt_p_ctx, const size_t n_unsent, const bool rewind_status);


    // --- upstream sampler-chain API adopted in this merge (decision #4). The four llama_sampler_init_dry/_grammar* declarations our side carried here are declared from upstream in the block above, with the generic llama_sampler* return the built TU actually uses. ---

    // Get list of built-in chat templates
    LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = llama_sampler_chain_default_params();
    //
    //    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    //
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        llama_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
    //
    //        ...
    //    }
    //
    //    llama_sampler_free(smpl);
    //

    typedef void * llama_sampler_context_t;

    struct llama_sampler_data {
        struct ggml_tensor * logits;
        struct ggml_tensor * probs;
        struct ggml_tensor * sampled;
        struct ggml_tensor * candidates;
    };

    // user code can implement the interface below in order to create custom llama_sampler
    struct llama_sampler_i {
        const char *           (*name)  (const struct llama_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct llama_sampler * smpl, llama_token token);              // can be NULL
        void                   (*apply) (      struct llama_sampler * smpl, llama_token_data_array * cur_p); // required
        void                   (*reset) (      struct llama_sampler * smpl);                                 // can be NULL
        struct llama_sampler * (*clone) (const struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL

        // [EXPERIMENTAL]
        // backend sampling interface:

        // return true if the backend supports all ops needed by the sampler and can handle up to n_outputs_max_per_seq outputs per sequence
        // note: call once per sampler
        bool (*backend_init)(
                struct llama_sampler       * smpl,
                ggml_backend_buffer_type_t   buft,
                uint32_t                     n_outputs_max_per_seq);

        // call after .backend_apply()
        void (*backend_accept)(
                struct llama_sampler * smpl,
                struct ggml_context  * ctx,
                struct ggml_cgraph   * gf,
                struct ggml_tensor   * selected_token);

        // call after .backend_init()
        void (*backend_apply)(
                struct llama_sampler      * smpl,
                struct ggml_context       * ctx,
                struct ggml_cgraph        * gf,
                struct llama_sampler_data * data);

        // called before graph execution to set inputs for the current ubatch
        void (*backend_set_input)(struct llama_sampler * smpl);

        // called before rebuilding a sampling graph to clear any internal sampler state
        void (*backend_reset)(struct llama_sampler * smpl);

        // copy mutable state from src into dst while keeping dst's references to the current sampling graph
        // src and dst must have the same type and configuration
        void (*copy_state)(const struct llama_sampler * src, struct llama_sampler * dst);
    };

    struct llama_sampler {
        struct llama_sampler_i * iface;

        llama_sampler_context_t ctx;
    };

    // [EXPERIMENTAL]
    // attach a sampler to the context
    // note: prefer initializing the context with llama_context_params.samplers when possible
    LLAMA_API bool llama_set_sampler(struct llama_context * ctx, llama_seq_id seq_id, struct llama_sampler * smpl);

    // mirror of llama_sampler_i:
    LLAMA_API struct llama_sampler * llama_sampler_init  (      struct llama_sampler_i * iface, llama_sampler_context_t ctx);
    LLAMA_API const char *           llama_sampler_name  (const struct llama_sampler * smpl);
    LLAMA_API void                   llama_sampler_accept(      struct llama_sampler * smpl, llama_token token);
    LLAMA_API void                   llama_sampler_apply (      struct llama_sampler * smpl, llama_token_data_array * cur_p);
    LLAMA_API void                   llama_sampler_reset (      struct llama_sampler * smpl);
    LLAMA_API struct llama_sampler * llama_sampler_clone (const struct llama_sampler * smpl);
    LLAMA_API void                   llama_sampler_copy  (const struct llama_sampler * src, struct llama_sampler * dst);
    // important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
    LLAMA_API void                   llama_sampler_free  (      struct llama_sampler * smpl);

    // llama_sampler_chain
    // a type of llama_sampler that can chain multiple samplers one after another

    LLAMA_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when llama_sampler_free is called
    LLAMA_API void                   llama_sampler_chain_add(      struct llama_sampler * chain, struct llama_sampler * smpl);

    // return NULL if:
    //   - the sampler is NULL
    //   - the sampler is not a llama_sampler_chain
    //   - the index is out of bounds, unless i == -1
    //   - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)
    LLAMA_API struct llama_sampler * llama_sampler_chain_get(      struct llama_sampler * chain, int32_t i);

    // the total number of samplers in the chain
    LLAMA_API int32_t                llama_sampler_chain_n  (const struct llama_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    LLAMA_API struct llama_sampler * llama_sampler_chain_remove(   struct llama_sampler * chain, int32_t i);

    // available samplers:

    LLAMA_API struct llama_sampler * llama_sampler_init_greedy(void);

    /// seed == LLAMA_DEFAULT_SEED to use a random seed.
    LLAMA_API struct llama_sampler * llama_sampler_init_dist(uint32_t seed);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    LLAMA_API struct llama_sampler * llama_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API struct llama_sampler * llama_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
    LLAMA_API struct llama_sampler * llama_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_API struct llama_sampler * llama_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    LLAMA_API struct llama_sampler * llama_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    LLAMA_API struct llama_sampler * llama_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    LLAMA_API struct llama_sampler * llama_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API llama_token llama_sample_token_mirostat(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   tau,
                           float   eta,
                         int32_t   m,
                           float * mu);

    // --- upstream sampler-chain API adopted in this merge (decision #4); the legacy llama_sample_* entry points above are KEPT and still have live callers ---
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API llama_token llama_sample_token_mirostat_v2(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   tau,
                           float   eta,
                           float * mu);

    /// @details Selects the token with the highest probability.
    ///          Does not compute the token probabilities. Use llama_sample_softmax() instead.
    LLAMA_API llama_token llama_sample_token_greedy(
            struct llama_context * ctx,
          llama_token_data_array * candidates);

    /// @details Randomly selects a token from the candidates based on their probabilities using the RNG of ctx.
    LLAMA_API llama_token llama_sample_token(
            struct llama_context * ctx,
          llama_token_data_array * candidates);

    // --- upstream sampler-chain API adopted in this merge (decision #4); the legacy llama_sample_* entry points above are KEPT and still have live callers ---
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    /// @details Initializes a GBNF grammar, see grammars/README.md for details.
    /// @param vocab The vocabulary that this grammar will be used with.
    /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    /// @param grammar_root The name of the start symbol for the grammar.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);

    DEPRECATED(LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root,
                         const char ** trigger_words,
                                size_t num_trigger_words,
                   const llama_token * trigger_tokens,
                                size_t num_trigger_tokens),
        "use llama_sampler_init_grammar_lazy_patterns instead");


    /// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy_patterns(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);


    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    LLAMA_API struct llama_sampler * llama_sampler_init_penalties(
                             int32_t   n_vocab,
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty)
                               float   penalty_repeat,   // must be > 0.0, 1.0 = disabled
                               float   penalty_freq,     // must be finite, 0.0 = disabled
                               float   penalty_present); // must be finite, 0.0 = disabled

    /// maximum number of distinct DRY sequence breakers, duplicates not counted
    /// processing each breaker scans the entire vocabulary, so the init cost is O(num_breakers x n_vocab)
    #define LLAMA_DRY_MAX_SEQ_BREAKERS 64

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    LLAMA_API struct llama_sampler * llama_sampler_init_dry(
            const struct llama_vocab *  vocab,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n, // last n tokens to penalize (0 = disable penalty)
                          const char ** seq_breakers,
                              size_t    num_breakers);

    /// adaptive-p: select tokens near a configurable target probability over time.
    ///
    /// the adaptive-p sampler transforms the token probability distribution to favor tokens
    /// that fall near a user-configurable probability target.
    ///
    /// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
    /// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
    /// adapted target probability at each sampling step, thus maintaining the desired target
    /// probability over time.
    ///
    /// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
    /// in the sampler chain (like mirostat, dist, greedy).
    ///
    /// only mild truncation before this sampler is recommended. we suggest applying min-p
    /// before adaptive-p as the only other active sampler in the chain.
    ///
    /// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    /// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
    /// @param seed   RNG seed
    ///
    /// ref: https://github.com/ggml-org/llama.cpp/pull/17927
    ///
    LLAMA_API struct llama_sampler * llama_sampler_init_adaptive_p(
                               float   target,
                               float   decay,
                            uint32_t   seed);

    /// @details FSD, Frustratingly Simple Decoding (Yang et al., arXiv:2305.12675): a deterministic anti-repetition decoder.
    /// score(v) = (1 - alpha) * p(v) - alpha * penalty(v), where p is the softmax of the `pool_size` highest logits, only the
    /// top `k` of them stay in the race, and penalty(v) is an interpolated, smoothed n-gram frequency of v as a follower of
    /// the last 0..n-1 accepted tokens (prompt + generated; the "anti language model"). The best score is selected, so FSD
    /// selects a token ID and must be last in the sampler chain (like dist and greedy). alpha = 0 is greedy within the pool.
    /// `random_first_n` > 0 selects Randomized FSD: for the first that many sampling steps, a fair coin chooses between
    /// FSD and ordinary top-p (0.9) / top-k (8) sampling (seeded by `seed`); 0 = plain FSD.
    /// History is learned from llama_sampler_accept(), so feed the prompt tokens through it before the first apply.
    /// @param pool_size candidates considered for the softmax (default in the paper's code: 50)
    /// @param k         candidates kept after the softmax (6)
    /// @param alpha     weight of the anti-LM penalty, 0..1 (0.5)
    /// @param n         n-gram order of the anti-LM, >= 1 (3)
    /// @param beta      back-off weight of the interpolation, 0..1 (0.9)
    LLAMA_API struct llama_sampler * llama_sampler_init_fsd(
                             int32_t   pool_size,
                             int32_t   k,
                               float   alpha,
                             int32_t   n,
                               float   beta,
                             int32_t   random_first_n,
                            uint32_t   seed);

    /// @details Constrain generation to an explicit allow-list of token ids.
    /// Any candidate whose id is not in `allowed` has its logit set to -INFINITY.
    /// This is a HARD constraint, unlike llama_sampler_init_logit_bias, which applies a finite
    /// additive bias that a sufficiently confident token can still overcome.
    /// An empty list (n_allowed <= 0 or allowed == NULL) is a no-op, NOT "forbid everything".
    /// Token ids outside [0, n_vocab) are ignored.
    LLAMA_API struct llama_sampler * llama_sampler_init_allowed_tokens(
                             int32_t   n_vocab,
                             int32_t   n_allowed,
                 const llama_token   * allowed);

    LLAMA_API struct llama_sampler * llama_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const llama_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    LLAMA_API struct llama_sampler * llama_sampler_init_infill(const struct llama_vocab * vocab);

    // Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
    LLAMA_API uint32_t llama_sampler_get_seed(const struct llama_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    // For multiple outputs from one sampler, call this function in output order without gaps.
    //
    // Shorthand for:
    //    const auto * logits = llama_get_logits_ith(ctx, idx);
    //    llama_token_data_array cur_p = { ... init from logits ... };
    //    llama_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    llama_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);

    // TODO: extend in the future
    //LLAMA_API void llama_decode_with_sampler(struct llama_context * ctx, struct llama_sampler * smpl, struct llama_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    LLAMA_API int32_t llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int32_t split_no, int32_t split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    LLAMA_API int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count);

    // Performance information
    LLAMA_API struct llama_timings llama_get_timings(struct llama_context * ctx);

    LLAMA_API void llama_print_timings(struct llama_context * ctx);
    LLAMA_API void llama_reset_timings(struct llama_context * ctx);

    // Print system information
    LLAMA_API const char * llama_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);

    LLAMA_API void llama_dump_timing_info_yaml(FILE * stream, const struct llama_context * ctx);

    //
    // MTP
    //

    enum llama_mtp_package {
        LLAMA_MTP_PACKAGE_NONE = 0,
        LLAMA_MTP_PACKAGE_EMBEDDED,
        LLAMA_MTP_PACKAGE_TARGET_ONLY,
        LLAMA_MTP_PACKAGE_COMPANION,
        LLAMA_MTP_PACKAGE_INVALID,
    };

    LLAMA_API int32_t llama_model_n_nextn_layer(const struct llama_model * model);

    LLAMA_API enum llama_mtp_package llama_model_mtp_package(const struct llama_model * model);

    LLAMA_API uint32_t llama_model_mtp_feature_width(const struct llama_model * model);

    LLAMA_API bool llama_model_step35_has_nextn_weights(const struct llama_model * model);

    // Set which, if any, MTP operation the context will use
    LLAMA_API void llama_set_mtp_op_type(struct llama_context * ctx, enum llama_mtp_op_type mtp_op_type);

    LLAMA_API void llama_set_draft_input_hidden_state(struct llama_context * ctx, const float * hidden_state);

    // S69 #31 EAGLE3 target 3-layer capture: enable on the TARGET ctx before its decode (lo/mid/hi layer ids),
    // then read the [3*n_embd] last-token concat after. Opt-in; default-OFF keeps the serving path identical.
    LLAMA_API void    llama_set_eagle3_capture(struct llama_context * ctx, bool enable, int lo, int mid, int hi);
    LLAMA_API int32_t llama_get_eagle3_target_feats(const struct llama_context * ctx, float * out);
    // Attach the target ctx to the draft ctx (draft reuses target tok_embd; records target n_embd).
    LLAMA_API void    llama_eagle3_set_target(struct llama_context * ctx_draft, const struct llama_context * ctx_target);
    LLAMA_API int32_t llama_eagle3_target_layers(const struct llama_context * ctx, int * out3);   // trained capture layers
    LLAMA_API int32_t llama_eagle3_get_d2t(const struct llama_context * ctx, int32_t * out);      // draft->target vocab map (i64/i32, as stored); out=NULL -> count; -1 = unsupported dtype
    // S69 Stage-1 API 3/4: embeddings_nextn equivalent (draft pre-norm hidden capture).
    LLAMA_API void    llama_set_eagle3_nextn_capture(struct llama_context * ctx, bool enable);
    LLAMA_API int32_t llama_get_eagle3_nextn_ith(const struct llama_context * ctx, int32_t i, float * out);
    // S69 Stage-2: per-position 3-layer feats getter. i<0 -> returns row count; i>=0 -> copies row i's
    // [3*n_embd_tgt] interleaved [lo|mid|hi] into out. Used by eagle3 process() to build per-position draft KV.
    LLAMA_API int32_t llama_eagle3_get_feats_pp_ith(const struct llama_context * ctx, int32_t i, float * out);

    LLAMA_API bool llama_reload_changed_tensors(struct llama_context * ctx);

    // --- adopted from upstream in this merge ---
    // The logger state is global so these functions are NOT thread safe.
    LLAMA_API void llama_log_get(ggml_log_callback * log_callback, void ** user_data);

    //
    // Performance utils
    //
    // NOTE: Used by llama.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
    //

    struct llama_perf_context_data {
        // ms == milliseconds
        double t_start_ms;  // absolute start time
        double t_load_ms;   // time needed for loading the model
        double t_p_eval_ms; // time needed for processing the prompt
        double t_eval_ms;   // time needed for generating tokens

        int32_t n_p_eval;   // number of prompt tokens
        int32_t n_eval;     // number of generated tokens
        int32_t n_reused;   // number of times a ggml compute graph had been reused
        int32_t n_reallocs_same; // number of graph reallocations during inference with an unchanged graph size
        int32_t n_reallocs_diff; // number of graph reallocations during inference with a changed graph size
    };

    struct llama_perf_sampler_data {
        double t_sample_ms; // time needed for sampling in ms

        int32_t n_sample;   // number of sampled tokens
    };

    LLAMA_API struct llama_perf_context_data llama_perf_context      (const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_print(const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_reset(      struct llama_context * ctx);

    // NOTE: the following work only with samplers constructed via llama_sampler_chain_init
    LLAMA_API struct llama_perf_sampler_data llama_perf_sampler      (const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_print(const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_reset(      struct llama_sampler * chain);

    // always returns true
    LLAMA_API bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    // RESTORED 2026-09-08, same drop pattern as llama_model_set_tensor_data_t above: upstream
    // declares it immediately before this struct (279b4bfaf^2:include/llama.h:1597), the merge
    // kept the struct that USES it and dropped the typedef. llama_opt_param_filter_all just above
    // already returns bool and matches this signature.
    typedef bool (*llama_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    struct llama_opt_params {
        uint32_t n_ctx_train; // assumed context size post training, use context size specified in llama_context if 0

        llama_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
        void * param_filter_ud;              // userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
        void * get_opt_pars_ud;                     // userdata for calculating optimizer parameters

        enum ggml_opt_optimizer_type optimizer_type;
    };

    LLAMA_API void llama_opt_init(struct llama_context * lctx, struct llama_model * model, struct llama_opt_params lopt_params);

    LLAMA_API void llama_opt_epoch(
            struct llama_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);

    //
    // In-Place TTT (row S69-FORK-INPLACE-TTT-2026-10-26-01; arXiv 2604.06169, src/llama-ttt.h)
    //

    // chunk length of a model with In-Place TTT fast weights; 0 for every other model
    LLAMA_API int32_t llama_model_ttt_chunk(const struct llama_model * model);

    // Speculative decoding on a TTT model: the most draft tokens that may follow the last accepted token at
    // position pos_last (the verify batch is [id_last @ pos_last, draft_1 @ pos_last + 1, ...]) so that no draft
    // token completes a chunk -- a rejected draft's fast-weight update could not be undone. INT32_MAX for a model
    // without TTT. Callers take min(their n_draft, this).
    LLAMA_API int32_t llama_ttt_draft_cap(const struct llama_model * model, llama_pos pos_last);

#ifdef __cplusplus
}
#endif

// Internal API to be implemented by llama.cpp and used by tests/benchmarks only
#ifdef LLAMA_API_INTERNAL

#include <random>
#include <string>
#include <vector>

struct ggml_tensor;

const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(
    struct llama_context * ctx
);


// Randomly selects a token from the candidates based on their probabilities using given std::mt19937.
// This is a temporary workaround in order to fix race conditions when sampling with multiple sequences.
llama_token llama_sample_token_with_rng(struct llama_context * ctx, llama_token_data_array * candidates, std::mt19937 & rng);

// Mirostat 1.0 / 2.0 drawing from the given RNG (the sampler's per-request seeded one) instead of the context's RNG, which a request
// seed never reaches; the plain llama_sample_token_mirostat[_v2] keep the context RNG for existing callers.
llama_token llama_sample_token_mirostat_with_rng(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, int32_t m,
                                                 float * mu, std::mt19937 & rng);
llama_token llama_sample_token_mirostat_v2_with_rng(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, float * mu,
                                                    std::mt19937 & rng);

// Randomly selects a token from the candidates following adaptive p sampler.
llama_token llama_sample_token_adaptive_p(struct llama_context * ctx, llama_token_data_array * candidates, struct llama_sampler_adaptive_p * adapt_p_ctx, std::mt19937 & rng);

#endif // LLAMA_API_INTERNAL

size_t llama_fill_from_utf8(void* utf8, void* cpts, void* scripts);

//
// upstream-name compatibility shims (C++ only)
//
// Upstream llama.cpp renamed and re-grouped part of its public API after our base diverged.
// Every name below already EXISTS here under an older spelling -- these are translations, not
// new capability, which is why they are header-only inline forwarders with no .cpp side and no
// ABI impact. They live outside the extern "C" block because two of them are C++ overloads.
//
// Kept here rather than in a separate header because upstream's mtmd sources include only
// "llama.h" and expect these names from it; a sync then stays a file copy.
//
#ifdef __cplusplus

// llama_model_rope_type: the fork's compat forwarder was REMOVED in the merge -- upstream now
// declares it at llama.h and defines it in its own TU, which is this shim's
// own documented exit condition.

// Upstream retargeted llama_tokenize from the model to the vocab. We have BOTH objects and a
// vocab-based entry point already (llama_vocab_tokenize), so this is an overload rather than a
// replacement -- the existing model-based llama_tokenize keeps working unchanged for every
// current caller in this tree.
static inline int32_t llama_tokenize(
        const struct llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special) {
    return llama_vocab_tokenize(vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special);
}

// llama_perf_context_print: the fork's compat forwarder was REMOVED in the merge -- upstream now
// declares it at llama.h and defines it in its own TU, which is this shim's
// own documented exit condition.

// Same retargeting for token_to_piece. Our fork already exposes the vocab-based entry point as
// llama_token_to_piece_vocab (include/llama.h:1275), so this too is a rename, not new code.
static inline int32_t llama_token_to_piece(
        const struct llama_vocab * vocab,
                     llama_token   token,
                            char * buf,
                         int32_t   length,
                         int32_t   lstrip,
                            bool   special) {
    return llama_token_to_piece_vocab(vocab, token, buf, length, lstrip, special);
}

#endif // __cplusplus

#endif // LLAMA_H
