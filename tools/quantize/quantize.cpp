#include "llama.h"

#include "build-info.h"
#include "common.h"
#include "imatrix-loader.h"
#include "imatrix-hessian.h"

#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <filesystem>

// result of parsing --tensor-type option
// changes to this struct must also be reflected in src/llama-quant.cpp
struct tensor_type_option {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
};

struct quant_option {
    std::string name;
    llama_ftype ftype;
    std::string desc;
};

static const std::vector<quant_option> QUANT_OPTIONS = {
    { "Q1_0",     LLAMA_FTYPE_MOSTLY_Q1_0,     " 1.125 bpw quantization",           },
    { "Q2_0",     LLAMA_FTYPE_MOSTLY_Q2_0,     " 2.25 bpw quantization (group 64)",  },
    { "Q4_0",     LLAMA_FTYPE_MOSTLY_Q4_0,     " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_1",     LLAMA_FTYPE_MOSTLY_Q4_1,     " 4.78G, +0.4511 ppl @ Llama-3-8B",  },
    { "MXFP4_MOE",LLAMA_FTYPE_MOSTLY_MXFP4_MOE," MXFP4 MoE",  },
    { "Q5_0",     LLAMA_FTYPE_MOSTLY_Q5_0,     " 5.21G, +0.1316 ppl @ Llama-3-8B",  },
    { "Q5_1",     LLAMA_FTYPE_MOSTLY_Q5_1,     " 5.65G, +0.1062 ppl @ Llama-3-8B",  },
    { "IQ2_XXS",  LLAMA_FTYPE_MOSTLY_IQ2_XXS,  " 2.06 bpw quantization",            },
    { "IQ2_XS",   LLAMA_FTYPE_MOSTLY_IQ2_XS,   " 2.31 bpw quantization",            },
    { "IQ2_S",    LLAMA_FTYPE_MOSTLY_IQ2_S,    " 2.5  bpw quantization",            },
    { "IQ2_M",    LLAMA_FTYPE_MOSTLY_IQ2_M,    " 2.7  bpw quantization",            },
    { "IQ1_S",    LLAMA_FTYPE_MOSTLY_IQ1_S,    " 1.56 bpw quantization",            },
    { "IQ1_M",    LLAMA_FTYPE_MOSTLY_IQ1_M,    " 1.75 bpw quantization",            },
    { "IQ1_XS",   LLAMA_FTYPE_MOSTLY_IQ1_XS,   " 1.44 bpw quantization (Unsloth narrow grid, needs imatrix)", },
    { "IQ1_XXS",  LLAMA_FTYPE_MOSTLY_IQ1_XXS,  " 1.31 bpw quantization (Unsloth narrow grid, needs imatrix)", },
    { "IQ1_XXXS", LLAMA_FTYPE_MOSTLY_IQ1_XXXS, " 1.19 bpw quantization (Unsloth narrow grid, needs imatrix)", },
    { "TQ1_0",    LLAMA_FTYPE_MOSTLY_TQ1_0,    " 1.69 bpw ternarization",           },
    { "TQ2_0",    LLAMA_FTYPE_MOSTLY_TQ2_0,    " 2.06 bpw ternarization",           },
    { "Q2_K",     LLAMA_FTYPE_MOSTLY_Q2_K,     " 2.96G, +3.5199 ppl @ Llama-3-8B",  },
    { "Q2_K_S",   LLAMA_FTYPE_MOSTLY_Q2_K_S,   " 2.96G, +3.1836 ppl @ Llama-3-8B",  },
    { "IQ3_XXS",  LLAMA_FTYPE_MOSTLY_IQ3_XXS,  " 3.06 bpw quantization",            },
    { "IQ3_S",    LLAMA_FTYPE_MOSTLY_IQ3_S,    " 3.44 bpw quantization",            },
    { "IQ3_M",    LLAMA_FTYPE_MOSTLY_IQ3_M,    " 3.66 bpw quantization mix",        },
    { "Q3_K",     LLAMA_FTYPE_MOSTLY_Q3_K_M,   "alias for Q3_K_M"                   },
    { "IQ3_XS",   LLAMA_FTYPE_MOSTLY_IQ3_XS,   " 3.3 bpw quantization",             },
    { "Q3_K_S",   LLAMA_FTYPE_MOSTLY_Q3_K_S,   " 3.41G, +1.6321 ppl @ Llama-3-8B",  },
    { "Q3_K_M",   LLAMA_FTYPE_MOSTLY_Q3_K_M,   " 3.74G, +0.6569 ppl @ Llama-3-8B",  },
    { "Q3_K_L",   LLAMA_FTYPE_MOSTLY_Q3_K_L,   " 4.03G, +0.5562 ppl @ Llama-3-8B",  },
    { "IQ4_NL",   LLAMA_FTYPE_MOSTLY_IQ4_NL,   " 4.50 bpw non-linear quantization", },
    { "IQ4_XS",   LLAMA_FTYPE_MOSTLY_IQ4_XS,   " 4.25 bpw non-linear quantization", },
    { "Q4_K",     LLAMA_FTYPE_MOSTLY_Q4_K_M,   "alias for Q4_K_M",                  },
    { "Q4_K_S",   LLAMA_FTYPE_MOSTLY_Q4_K_S,   " 4.37G, +0.2689 ppl @ Llama-3-8B",  },
    { "Q4_K_M",   LLAMA_FTYPE_MOSTLY_Q4_K_M,   " 4.58G, +0.1754 ppl @ Llama-3-8B",  },
    { "Q5_K",     LLAMA_FTYPE_MOSTLY_Q5_K_M,   "alias for Q5_K_M",                  },
    { "Q5_K_S",   LLAMA_FTYPE_MOSTLY_Q5_K_S,   " 5.21G, +0.1049 ppl @ Llama-3-8B",  },
    { "Q5_K_M",   LLAMA_FTYPE_MOSTLY_Q5_K_M,   " 5.33G, +0.0569 ppl @ Llama-3-8B",  },
    { "Q6_K",     LLAMA_FTYPE_MOSTLY_Q6_K,     " 6.14G, +0.0217 ppl @ Llama-3-8B",  },
    { "Q8_0",     LLAMA_FTYPE_MOSTLY_Q8_0,     " 7.96G, +0.0026 ppl @ Llama-3-8B",  },
    { "F16",      LLAMA_FTYPE_MOSTLY_F16,      "14.00G, +0.0020 ppl @ Mistral-7B",  },
    { "BF16",     LLAMA_FTYPE_MOSTLY_BF16,     "14.00G, -0.0050 ppl @ Mistral-7B",  },
    // ---- restored 2026-09-08: this fork's own ftypes, dropped when the merge took
    // upstream's quantize CLI wholesale. examples/quantize/ (which held them) is no longer
    // entered by any add_subdirectory, so these 55 names could no longer be selected from
    // the command line -- by NAME or by NUMBER, since the numeric path validates against
    // this same table. Every other link of the chain survived: the ftypes are in llama.h,
    // the ftype->type mapping is in llama-quantize.cpp and the ggml_quantize_chunk cases
    // are in ggml.c. Rows already present upstream are NOT duplicated here; upstream's
    // copy is kept because it carries refreshed Llama-3-8B measurements.
    { "Q6_0",     LLAMA_FTYPE_MOSTLY_Q6_0,     " 6.5 bpw quantization",             },
    { "Q1_0_G128",LLAMA_FTYPE_MOSTLY_Q1_0_G128," 1.125 bpw Bonsai ternary, group 128",},
    { "Q2_0_G128",LLAMA_FTYPE_MOSTLY_Q2_0_G128," 2.125 bpw Bonsai ternary, group 128",},
    { "Q1_0_G128_R8",LLAMA_FTYPE_MOSTLY_Q1_0_G128_R8," Q1_0_G128 repacked 8 rows (ik_llama #2541)",},
    // Prism / Bonsai-2 (ik_llama #2550). PQ2_0 is byte-identical to Q2_0_G128; both ids are kept so files from either tool load.
    { "PQ2_0",    LLAMA_FTYPE_MOSTLY_PQ2_0,    " 2.13 bpw ternary, group-128 (Prism)",},
    { "PTQ1_0",   LLAMA_FTYPE_MOSTLY_PTQ1_0,   " 1.75 bpw ternary, base-3 (Prism)",},
    { "PQ2_0_R8", LLAMA_FTYPE_MOSTLY_PQ2_0_R8, " PQ2_0 repacked 8 rows",},
    { "PTQ1_0_R8",LLAMA_FTYPE_MOSTLY_PTQ1_0_R8," PTQ1_0 repacked 8 rows",},
    { "MXFP4",    LLAMA_FTYPE_MOSTLY_MXFP4,    " 4.25 bpw 4-bit float quantization",},
    { "MXFP6_E2M3", LLAMA_FTYPE_MOSTLY_MXFP6_E2M3, " 6.25 bpw OCP MX FP6 (E2M3), E8M0 scale per 32",},
    { "MXFP6_E3M2", LLAMA_FTYPE_MOSTLY_MXFP6_E3M2, " 6.25 bpw OCP MX FP6 (E3M2), E8M0 scale per 32",},
    { "MXFP8_E4M3", LLAMA_FTYPE_MOSTLY_MXFP8_E4M3, " 8.25 bpw OCP MX FP8 (E4M3), E8M0 scale per 32",},
    { "MXFP8_E5M2", LLAMA_FTYPE_MOSTLY_MXFP8_E5M2, " 8.25 bpw OCP MX FP8 (E5M2), E8M0 scale per 32",},
    { "TQ3_1S",   LLAMA_FTYPE_MOSTLY_TQ3_1S,   " 4.00 bpw TurboQuant WHT-rotated 3-bit Lloyd-Max, 2 fp16 scales per 32",},
    { "TQ4_1S",   LLAMA_FTYPE_MOSTLY_TQ4_1S,   " 5.00 bpw TurboQuant WHT-rotated 4-bit Lloyd-Max, 2 fp16 scales per 32",},
    { "NVFP4",    LLAMA_FTYPE_MOSTLY_NVFP4,    " 4.5 bpw 4-bit float, UE4M3 scale per 16",},
    { "F8_E4M3",  LLAMA_FTYPE_MOSTLY_F8_E4M3,  " 8 bpw OCP FP8 E4M3 (unscaled)",},
    { "NVFP4_KS", LLAMA_FTYPE_MOSTLY_NVFP4_KS, " 4.5 bpw NVFP4 blocks with one fp32 scale per row (ik)",},
    { "MXFP4_R8", LLAMA_FTYPE_MOSTLY_MXFP4_R8, " MXFP4 repacked",                   },
    { "IQ2_XXS_R4",LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4,"IQ2_XXS repacked",            },
    { "IQ2_XS_R4",LLAMA_FTYPE_MOSTLY_IQ2_XS_R4,"IQ2_XS repacked",            },
    { "IQ2_M_R4", LLAMA_FTYPE_MOSTLY_IQ2_M_R4, " 2.7  bpw quantization",            },
    { "IQ1_S_R4", LLAMA_FTYPE_MOSTLY_IQ1_S_R4, " 1.5 bpw quantization",             },
    { "IQ1_M_R4", LLAMA_FTYPE_MOSTLY_IQ1_M_R4, " 1.75 bpw quantization",            },
    { "IQ1_BN",   LLAMA_FTYPE_MOSTLY_IQ1_BN,   " 1.62 bpw quantization (Bitnet)",   },
    { "IQ2_BN",   LLAMA_FTYPE_MOSTLY_IQ2_BN,   " 2.00 bpw quantization (Bitnet)",   },
    { "IQ2_BN_R4",LLAMA_FTYPE_MOSTLY_IQ2_BN_R4," 2.00 bpw quantization (Bitnet)",   },
    { "Q2_K_R4",  LLAMA_FTYPE_MOSTLY_Q2_K_R4,  "Q2_K_S repacked", },
    { "IQ3_KT",   LLAMA_FTYPE_MOSTLY_IQ3_KT,   " 3.125 bpw trellis quantization",   },
    { "IQ4_KT",   LLAMA_FTYPE_MOSTLY_IQ4_KT,   " 4.0 bpw trellis quantization",     },
    { "IQ3_XXS_R4",LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4,"IQ3_XXS repacked",            },
    { "IQ3_S_R4", LLAMA_FTYPE_MOSTLY_IQ3_S_R4, "IQ3_S repacked",            },
    { "Q3_K_R4",  LLAMA_FTYPE_MOSTLY_Q3_K_R4,  "Q3_K_S repacked" },
    { "IQ4_NL_R4",LLAMA_FTYPE_MOSTLY_IQ4_NL_R4," 4.50 bpw non-linear quantization", },
    { "IQ4_XS_R8",LLAMA_FTYPE_MOSTLY_IQ4_XS_R8," 4.25 bpw non-linear quantization", },
    { "Q4_0_R8",  LLAMA_FTYPE_MOSTLY_Q4_0_R8,  " 4.50 bpw quantization",            },
    { "Q2_0_R8",  LLAMA_FTYPE_MOSTLY_Q2_0_R8,  " 2.25 bpw 8-row repacked",          },
    { "Q5_0_R4",  LLAMA_FTYPE_MOSTLY_Q5_0_R4,  " 5.50 bpw quantization",            },
    { "Q6_0_R4",  LLAMA_FTYPE_MOSTLY_Q6_0_R4,  " 6.50 bpw quantization",            },
    { "Q8_0_R8",  LLAMA_FTYPE_MOSTLY_Q8_0_R8,  " 8.50 bpw quantization",            },
    { "Q8_KV",    LLAMA_FTYPE_MOSTLY_Q8_KV,    " 8.00 bpw quantization",            },
    { "IQ4_KS",   LLAMA_FTYPE_MOSTLY_IQ4_KS,   " 4.25 bpw non-linear quantization", },
    { "IQ4_KS_R4",LLAMA_FTYPE_MOSTLY_IQ4_KS_R4,"IQ4_KS repacked", },
    { "IQ4_KS_R16",LLAMA_FTYPE_MOSTLY_IQ4_KS_R16,"IQ4_KS 16-row repacked", },
    { "IQ5_KS_R16",LLAMA_FTYPE_MOSTLY_IQ5_KS_R16,"IQ5_KS 16-row repacked", },
    { "IQ3_KS_R16",LLAMA_FTYPE_MOSTLY_IQ3_KS_R16,"IQ3_KS 16-row repacked", },
    { "IQ5_KS_R4",LLAMA_FTYPE_MOSTLY_IQ5_KS_R4,"IQ5_KS repacked", },
    { "IQ4_KSS",  LLAMA_FTYPE_MOSTLY_IQ4_KSS,  " 4.0 bpw non-linear quantization",  },
    { "IQ5_KS",   LLAMA_FTYPE_MOSTLY_IQ5_KS,   " 5.25 bpw non-linear quantization", },
    { "IQ2_K",    LLAMA_FTYPE_MOSTLY_IQ2_K,    " 2.375 bpw non-linear quantization",},
    { "IQ2_K_R4", LLAMA_FTYPE_MOSTLY_IQ2_K_R4, "IQ2_K repacked",},
    { "IQ2_KS",   LLAMA_FTYPE_MOSTLY_IQ2_KS,   " 2.1875 bpw non-linear quantization",},
    { "IQ1_KT",   LLAMA_FTYPE_MOSTLY_IQ1_KT,   " 1.75 bpw trellis quantization",   },
    { "IQ2_KT",   LLAMA_FTYPE_MOSTLY_IQ2_KT,   " 2.125 bpw trellis quantization",   },
    { "IQ2_KL",   LLAMA_FTYPE_MOSTLY_IQ2_KL,   " 2.69 bpw non-linear quantization", },
    { "IQ3_KS",   LLAMA_FTYPE_MOSTLY_IQ3_KS,   " 3.19 bpw non-linear quantization", },
    { "IQ3_K",    LLAMA_FTYPE_MOSTLY_IQ3_K,    " 3.44 bpw non-linear quantization", },
    { "IQ3_K_R4", LLAMA_FTYPE_MOSTLY_IQ3_K_R4, "IQ3_K repacked", },
    { "IQ3_KL",   LLAMA_FTYPE_MOSTLY_IQ3_KL,   " 4 bpw non-linear quantization mix",},
    { "IQ4_K",    LLAMA_FTYPE_MOSTLY_IQ4_K,    " 4.5 bpw non-linear quantization",  },
    { "IQ4_K_R4", LLAMA_FTYPE_MOSTLY_IQ4_K_R4, "IQ4_K repacked",  },
    { "IQ5_K",    LLAMA_FTYPE_MOSTLY_IQ5_K,    " 5.5 bpw non-linear quantization",  },
    { "IQ5_K_R4", LLAMA_FTYPE_MOSTLY_IQ5_K_R4, "IQ5_K repacked",  },
    { "IQ6_K",    LLAMA_FTYPE_MOSTLY_IQ6_K,    " 6.6 bpw non-linear quantization",  },
    { "Q4_K_R4",  LLAMA_FTYPE_MOSTLY_Q4_K_R4,  "Q4_K_S repacked", },
    { "Q5_K_R4",  LLAMA_FTYPE_MOSTLY_Q5_K_R4,  "Q5_K_S repacked", },
    { "Q6_K_R4",  LLAMA_FTYPE_MOSTLY_Q6_K_R4,  "Q6_K repacked", },
    { "Q8_K_R8",  LLAMA_FTYPE_MOSTLY_Q8_K_R8,  "Q8_K repacked", },
    { "Q8_K_R16", LLAMA_FTYPE_MOSTLY_Q8_K_R16, "Q8_K repacked", },
    { "Q8_KV_R8", LLAMA_FTYPE_MOSTLY_Q8_KV_R8, "Q8_KV repacked", },
    { "Q4_0_4_4", LLAMA_FTYPE_MOSTLY_Q4_0_4_4, " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_0_4_8", LLAMA_FTYPE_MOSTLY_Q4_0_4_8, " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_0_8_8", LLAMA_FTYPE_MOSTLY_Q4_0_8_8, " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "BF16_R16", LLAMA_FTYPE_MOSTLY_BF16_R16, "14.00G, -0.0050 ppl @ Mistral-7B", },
    { "F32",      LLAMA_FTYPE_ALL_F32,         "26.00G              @ 7B",          },
    // Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
    { "COPY",     LLAMA_FTYPE_ALL_F32,         "only copy tensors, no quantizing",  },
};

static const char * const LLM_KV_QUANTIZE_IMATRIX_FILE       = "quantize.imatrix.file";
static const char * const LLM_KV_QUANTIZE_IMATRIX_DATASET    = "quantize.imatrix.dataset";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES  = "quantize.imatrix.entries_count";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS   = "quantize.imatrix.chunks_count";

static bool striequals(const char * a, const char * b) {
    while (*a && *b) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
        a++; b++;
    }
    return *a == *b;
}

static bool try_parse_ftype(const std::string & ftype_str_in, llama_ftype & ftype, std::string & ftype_str_out) {
    std::string ftype_str;

    for (auto ch : ftype_str_in) {
        ftype_str.push_back(std::toupper(ch));
    }
    for (const auto & it : QUANT_OPTIONS) {
        if (striequals(it.name.c_str(), ftype_str.c_str())) {
            ftype = it.ftype;
            ftype_str_out = it.name;
            return true;
        }
    }
    try {
        int ftype_int = std::stoi(ftype_str);
        for (const auto & it : QUANT_OPTIONS) {
            if (it.ftype == ftype_int) {
                ftype = it.ftype;
                ftype_str_out = it.name;
                return true;
            }
        }
    }
    catch (...) {
        // stoi failed
    }
    return false;
}

// Parse a common-style CUDA list (CUDA0,CUDA1, bare index ok) into ordinals.
// From ik's examples/quantize copy (#2674), which this tree does not build.
static bool parse_cuda_device_list(const std::string & arg, std::vector<int> & devs) {
    std::vector<int> out;
    for (const auto & item : string_split<std::string>(arg, ',')) {
        size_t beg = item.find_first_not_of(" \t");
        size_t end = item.find_last_not_of(" \t");
        if (beg == std::string::npos) {
            return false;
        }
        std::string first = item.substr(beg, end - beg + 1);
        std::string digits = first;
        if (first.size() > 4 && (first.compare(0, 4, "CUDA") == 0 || first.compare(0, 4, "cuda") == 0)) {
            digits = first.substr(4);
        }
        if (digits.empty()) {
            return false;
        }
        for (char c : digits) {
            if (!isdigit((unsigned char) c)) {
                return false;
            }
        }
        try {
            out.push_back(std::stoi(digits));
        }
        catch (const std::exception &) {
            return false;
        }
    }
    if (out.empty()) {
        return false;
    }
    devs = std::move(out);
    return true;
}

[[noreturn]]
static void usage(const char * executable) {
    printf("usage: %s [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights]\n", executable);
    printf("       [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--tensor-type-file]\n");
    printf("       [--custom-q] [--attn-*-type] [--ffn-*-type] [--extra-output-tensor] [--repack] [--repack-pattern]\n");
    printf("       [--symmetric-q40] [--slow-iq2ks] [--hide-imatrix] [--ignore-imatrix-rules] [--partial-requant]\n");
    printf("       [--fudge-factors] [--prune-layers] [--keep-split] [--individual-tensors] [--skip-first-shard] [--override-kv] [--dry-run] [--nvfp4-scale auto|one|two]\n");
    printf("       [--slab-size N] [--cuda-quantize] [-dev/--device DEVICES] [-ts/--tensor-split SPLIT]\n");
    printf("       [--gptq-hessian FILE] [--gptq-damp F] [--gptq-no-propagate]\n");
    printf("       [--max-buffer-size]\n");
    printf("       model-f32.gguf [model-quant.gguf] type [nthreads]\n\n");
    printf("  --allow-requantize\n");
    printf("                                      allow requantizing tensors that have already been quantized\n");
    printf("                                      WARNING: this can severely reduce quality compared to quantizing\n");
    printf("                                               from 16bit or 32bit!\n");
    printf("  --leave-output-tensor\n");
    printf("                                      leave output.weight un(re)quantized\n");
    printf("                                      increases model size but may also increase quality, especially when requantizing\n");
    printf("  --pure\n");
    printf("                                      disable k-quant mixtures and quantize all tensors to the same type\n");
    printf("  --gptq-hessian file_name\n");
    printf("                                      GPTQ error-compensated rounding from the block-diagonal Hessian sidecar of\n");
    printf("                                      llama-imatrix --hessian-block N (N >= 2x the quant block: 64 for Q4_0, 512 for k-quants)\n");
    printf("  --gptq-damp F\n");
    printf("                                      Hessian dampening, H_ii += F * mean(diag H) (default 0.01)\n");
    printf("  --gptq-no-propagate\n");
    printf("                                      control arm: the same block-wise quantisation with no error propagation\n");
    printf("  --imatrix file_name\n");
    printf("                                      use data in file_name as importance matrix for quant optimizations\n");
    printf("  --include-weights tensor_name\n");
    printf("                                      use importance matrix for this/these tensor(s)\n");
    printf("  --exclude-weights tensor_name\n");
    printf("                                      do not use importance matrix for this/these tensor(s)\n");
    printf("  --output-tensor-type ggml_type\n");
    printf("                                      use this ggml_type for the output.weight tensor\n");
    printf("  --token-embedding-type ggml_type\n");
    printf("                                      use this ggml_type for the token embeddings tensor\n");
    printf("  --tensor-type tensor_name=ggml_type\n");
    printf("                                      quantize this tensor to this ggml_type\n");
    printf("                                      this is an advanced option to selectively quantize tensors. may be specified multiple times.\n");
    printf("                                      example: --tensor-type attn_q=q8_0\n");
    printf("  --tensor-type-file tensor_types.txt\n");
    printf("                                      list of tensors to quantize to a specific ggml_type\n");
    printf("                                      this is an advanced option to selectively quantize a long list of tensors.\n");
    printf("                                      the file should use the same format as above, separated by spaces or newlines.\n");
    printf("  --custom-q regex1=type1,regex2=type2...\n");
    printf("                                      custom quantization type rules; the first regex that matches a tensor name wins\n");
    printf("  --attn-q-type / --attn-k-type / --attn-v-type / --attn-qkv-type / --attn-output-type ggml_type\n");
    printf("  --ffn-gate-type / --ffn-down-type / --ffn-up-type / --ffn-gate-inp-type ggml_type\n");
    printf("  --per-layer-token-embedding-type ggml_type\n");
    printf("                                      use this ggml_type for those tensors\n");
    printf("  --extra-output-tensor ggml_type\n");
    printf("                                      requantize and add an output tensor of that type\n");
    printf("  --repack\n");
    printf("                                      only repack tensors to the corresponding _r4/_r8 variant where one exists\n");
    printf("  --repack-pattern regex1,regex2...\n");
    printf("                                      repack only tensors whose name matches one of these regexes\n");
    printf("  --symmetric-q40\n");
    printf("                                      use the [-7:7] range for Q4_0 (turns off imatrix)\n");
    printf("  --slow-iq2ks\n");
    printf("                                      use the original, very slow IQ2_KS quantization method\n");
    printf("  --hide-imatrix\n");
    printf("                                      do not store imatrix file/dataset details in the quantized model\n");
    printf("  --ignore-imatrix-rules\n");
    printf("                                      quantize to types that normally refuse to run without an imatrix\n");
    printf("  --partial-requant\n");
    printf("                                      with split output: quantize only the split files missing from the destination\n");
    printf("  --fudge-factors type1=ff1,type2=ff2...\n");
    printf("                                      multiply the block scales of these quantization types by these factors\n");
    printf("                                      (defaults: ggml_get_quantize_fudge_factor; e.g. iq2_xs=1.05, most types 1.0)\n");
    printf("  --prune-layers L0,L1,L2...\n");
    printf("                                      comma-separated list of layer numbers to prune from the model\n");
    printf("                                      WARNING: this is an advanced option, use with care.\n");
    printf("  --nvfp4-scale auto|one|two\n");
    printf("                                      NVFP4 scale layout: one = per-16 UE4M3 block scale only (historic; underflows on small-magnitude\n");
    printf("                                      weights), two = block scale plus one FP32 per-tensor scale written as <name>.scale,\n");
    printf("                                      auto (default) = two for architectures whose graph applies it, one otherwise\n");
    printf("  --slab-size N\n");
    printf("                                      process tensors larger than N MiB of f32 in slabs of up to N MiB, or of one\n");
    printf("                                      expert slice or row group if that is larger (default: 1024, 0 = never)\n");
    printf("  --cuda-quantize\n");
    printf("                                      quantize IQ4_KT, IQ3_KT, IQ2_KT, IQ1_KT, IQ3_KS, IQ4_KS, IQ4_KS_R4, IQ4_KS_R16, IQ5_KS\n");
    printf("                                      IQ5_KS_R4 and IQ5_KS_R16 tensors on a CUDA device; other types use the CPU\n");
    printf("                                      (a build without CUDA warns and quantizes everything on the CPU)\n");
    printf("  -dev DEVICES, --device DEVICES\n");
    printf("                                      CUDA device(s) for --cuda-quantize (default CUDA0; 2+ devices = row-split, e.g. -dev CUDA0,CUDA1)\n");
    printf("  -ts SPLIT, --tensor-split SPLIT\n");
    printf("                                      row-split shares for 2+ GPUs (e.g. -ts 3,3,2; equal when absent; needs -dev with 2+ devices)\n");
    printf("  --keep-split\n");
    printf("                                      generate quantized model in the same shards as input\n");
    printf("  --individual-tensors N1,N2,...  (alias --only-splits)\n");
    printf("                                      with --keep-split: open and write only shard 1 and these shards (1-based, as in\n");
    printf("                                      -0000N-of-0000M.gguf); the other input shards need not exist. Each written shard\n");
    printf("                                      equals the same shard of a full run when the types do not depend on tensor\n");
    printf("                                      position (--pure, --custom-q / --tensor-type for every tensor)\n");
    printf("  --skip-first-shard\n");
    printf("                                      with --individual-tensors: do not write shard 1 (the metadata shard)\n");
    printf("  --override-kv KEY=TYPE:VALUE\n");
    printf("                                      override model metadata by key in the quantized model. may be specified multiple times.\n");
    printf("                                      WARNING: this is an advanced option, use with care.\n");
    printf("  --dry-run\n");
    printf("                                      calculate and show the final quantization size without performing quantization\n");
    printf("                                      example: llama-quantize --dry-run model-f32.gguf Q4_K\n");
    printf("  --max-buffer-size MiB\n");
    printf("                                      max amount of tensor rows kept in memory while quantizing one tensor (default: 8192)\n");
    printf("                                      lower it to quantize models with very large tensors on a machine with little RAM\n\n");
    printf("note: --include-weights and --exclude-weights cannot be used together\n\n");
    printf("-----------------------------------------------------------------------------\n");
    printf(" allowed quantization types\n");
    printf("-----------------------------------------------------------------------------\n\n");
    for (const auto & it : QUANT_OPTIONS) {
        if (it.name != "COPY") {
            printf("  %2d  or  ", it.ftype);
        } else {
            printf("          ");
        }
        printf("%-7s : %s\n", it.name.c_str(), it.desc.c_str());
    }
    exit(1);
}

static int load_imatrix(const std::string & imatrix_file, std::vector<std::string> & imatrix_datasets, std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    common_imatrix loaded;
    if (!common_imatrix_load(imatrix_file, loaded)) {
        fprintf(stderr, "%s: failed to load imatrix from '%s'\n", __func__, imatrix_file.c_str());
        exit(1);
    }

    if (!loaded.is_legacy && !loaded.has_metadata) {
        fprintf(stderr, "%s: missing imatrix metadata in file %s\n", __func__, imatrix_file.c_str());
        exit(1);
    }

    for (const auto & [name, entry] : loaded.entries) {
        auto & e = imatrix_data[name];
        e.resize(entry.sums.size());

        if (!loaded.is_legacy) {
            // GGUF format: normalize by per-expert counts
            const int64_t ncounts = entry.counts.size();
            const int64_t ne0     = (int64_t) entry.sums.size() / ncounts;

            for (int64_t j = 0; j < ncounts; ++j) {
                const float count = (float) entry.counts[j];
                if (count > 0.0f) {
                    for (int64_t i = 0; i < ne0; ++i) {
                        e[j*ne0 + i] = entry.sums[j*ne0 + i] / count;
                    }
                } else {
                    for (int64_t i = 0; i < ne0; ++i) {
                        e[j*ne0 + i] = 1;
                    }
                }
            }

            if (getenv("LLAMA_TRACE")) {
                float max_count = 0.0f;
                for (int64_t j = 0; j < ncounts; ++j) {
                    const float count = (float) entry.counts[j];
                    if (count > max_count) {
                        max_count = count;
                    }
                }
                printf("%s: loaded data (size = %6d, n_tokens = %6d, n_chunks = %6d) for '%s'\n",
                       __func__, int(e.size()), int(max_count), int(max_count / loaded.chunk_size), name.c_str());
            }
        } else {
            // Legacy format: sums contain (raw/count)*ncall, divide by ncall
            const int64_t ncall = entry.counts.empty() ? 0 : entry.counts[0];
            if (ncall > 0) {
                for (size_t i = 0; i < entry.sums.size(); ++i) {
                    e[i] = entry.sums[i] / ncall;
                }
            } else {
                for (size_t i = 0; i < entry.sums.size(); ++i) {
                    e[i] = entry.sums[i];
                }
            }

            if (getenv("LLAMA_TRACE")) {
                printf("%s: loaded data (size = %6d, ncall = %6d) for '%s'\n",
                       __func__, int(e.size()), int(ncall), name.c_str());
            }
        }
    }

    imatrix_datasets = std::move(loaded.datasets);

    if (!imatrix_datasets.empty()) {
        printf("%s: imatrix datasets=['%s'", __func__, imatrix_datasets[0].c_str());
        for (size_t i = 1; i < imatrix_datasets.size(); ++i) {
            printf(", '%s'", imatrix_datasets[i].c_str());
        }
        printf("]\n");
    }

    printf("%s: loaded %d importance matrix entries from %s computed on %d chunks\n", __func__, int(imatrix_data.size()), imatrix_file.c_str(), loaded.chunk_count);

    return loaded.chunk_count;
}

static int prepare_imatrix(const std::string & imatrix_file,
        std::vector<std::string> & imatrix_dataset,
        const std::vector<std::string> & included_weights,
        const std::vector<std::string> & excluded_weights,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    int m_last_call = -1;
    if (!imatrix_file.empty()) {
        m_last_call = load_imatrix(imatrix_file, imatrix_dataset, imatrix_data);
    }
    if (imatrix_data.empty()) {
        return m_last_call;
    }
    if (!excluded_weights.empty()) {
        for (const auto & name : excluded_weights) {
            for (auto it = imatrix_data.begin(); it != imatrix_data.end();) {
                auto pos = it->first.find(name);
                if (pos != std::string::npos) {
                    it = imatrix_data.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    if (!included_weights.empty()) {
        std::unordered_map<std::string, std::vector<float>> tmp;
        for (const auto & name : included_weights) {
            for (auto & e : imatrix_data) {
                auto pos = e.first.find(name);
                if (pos != std::string::npos) {
                    tmp.emplace(std::move(e));
                }
            }
        }
        imatrix_data = std::move(tmp);
    }
    if (!imatrix_data.empty()) {
        printf("%s: have %d importance matrix entries\n", __func__, int(imatrix_data.size()));
    }
    return m_last_call;
}

static ggml_type parse_ggml_type(const char * arg) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        auto type = (ggml_type)i;
        const auto * name = ggml_type_name(type);
        if (name && striequals(name, arg)) {
            return type;
        }
    }
    fprintf(stderr, "\n%s: invalid ggml_type '%s'\n\n", __func__, arg);
    return GGML_TYPE_COUNT;
}

static bool parse_tensor_type(const char * data, std::vector<tensor_type_option> & tensor_type) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr) {
        printf("\n%s: malformed tensor type '%s'\n\n", __func__, data);
        return false;
    }

    const size_t tn_len = sep - data;
    if (tn_len == 0) {
        printf("\n%s: missing tensor name\n\n", __func__);
        return false;
    }
    if (const size_t qt_len = strlen(sep); qt_len == 1) {
        printf("\n%s: missing quantization type\n\n", __func__);
        return false;
    }

    std::string tn(data, tn_len);
    std::transform(tn.begin(), tn.end(), tn.begin(), tolower);
    sep++;
    tensor_type_option tensor_type_opt;
    tensor_type_opt.name = tn;
    tensor_type_opt.type = parse_ggml_type(sep);
    tensor_type.emplace_back(std::move(tensor_type_opt));
    if (tensor_type_opt.type == GGML_TYPE_COUNT) {
        printf("\n%s: invalid quantization type '%s'\n\n", __func__, sep);
        return false;
    }

    return true;
}

static bool parse_tensor_type_file(const char * filename, std::vector<tensor_type_option> & tensor_type) {
    std::ifstream file(filename);
    if (!file) {
        printf("\n%s: failed to open file '%s': %s\n\n", __func__, filename, std::strerror(errno));
        return false;
    }

    std::string arg;
    while (file >> arg) {
        if (!parse_tensor_type(arg.c_str(), tensor_type)) {
            return false;
        }
    }

    return true;
}

// fork --custom-q regex1=type1,...: must match the type src/llama-quantize.cpp casts params.custom_quants to
using CustomQ = std::pair<std::string, ggml_type>;

static bool parse_custom_quants(const char * data, std::vector<CustomQ> & custom_quants) {
    for (const auto & item : string_split<std::string>(data, ',')) {
        const auto pos = item.find('=');
        if (pos == std::string::npos || pos == 0) {
            printf("\n%s: malformed custom quantization rule '%s' (expected regex=type)\n\n", __func__, item.c_str());
            return false;
        }
        const ggml_type type = parse_ggml_type(item.substr(pos + 1).c_str());
        if (type == GGML_TYPE_COUNT) {
            return false;
        }
        printf("Adding custom rule %s -> %s\n", item.substr(0, pos).c_str(), ggml_type_name(type));
        custom_quants.emplace_back(item.substr(0, pos), type);
    }
    return true;
}

// upstream ik_llama --fudge-factors type1=ff1,type2=ff2: per-type ggml_set_quantize_fudge_factor
static bool parse_fudge_factors(const char * data, std::unordered_map<ggml_type, float> & factors) {
    for (const auto & item : string_split<std::string>(data, ',')) {
        const auto pos = item.find('=');
        if (pos == std::string::npos) {
            printf("\n%s: malformed fudge factor '%s' (expected type=factor)\n\n", __func__, item.c_str());
            return false;
        }
        const std::string type_str = item.substr(0, pos);
        const std::string val_str  = item.substr(pos + 1);
        const ggml_type type = parse_ggml_type(type_str.c_str());
        if (type == GGML_TYPE_COUNT) {
            return false;
        }
        // strtof + end check so "1.05x" or "" fails instead of being read as a prefix; a factor
        // <= 0 or non-finite would zero or corrupt every block scale of that type
        char * end = nullptr;
        const float val = strtof(val_str.c_str(), &end);
        if (val_str.empty() || *end != '\0' || !std::isfinite(val) || val <= 0.0f) {
            printf("\n%s: invalid fudge factor '%s' for type %s\n\n", __func__, val_str.c_str(), ggml_type_name(type));
            return false;
        }
        factors[type] = val;
        printf("Adding scale fudge factor = %g for type %s (default %g)\n", val, ggml_type_name(type), ggml_get_quantize_fudge_factor(type));
    }
    return true;
}

static bool parse_layer_prune(const char * data, std::vector<int> & prune_layers) {
    if (!data) {
        printf("\n%s: no layer pruning ids provided\n\n", __func__);
        return false;
    }

    const auto block_ids = string_split<std::string>(data, ',');
    for (const auto & block_id : block_ids) {
        int id;
        try {
            id = std::stoi(block_id);
        } catch (...) {
            id = -1;
        }
        if (id < 0) {
            printf("\n%s: invalid layer id '%s'\n\n", __func__, block_id.c_str());
            return false;
        }
        prune_layers.emplace_back(id);
    }

    sort(prune_layers.begin(), prune_layers.end());
    prune_layers.erase(std::unique(prune_layers.begin(), prune_layers.end()), prune_layers.end());
    return true;
}

// satisfies -Wmissing-declarations
int llama_quantize(int argc, char ** argv);

int llama_quantize(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 3) {
        usage(argv[0]);
    }

    llama_model_quantize_params params = llama_model_quantize_default_params();

    int arg_idx = 1;
    std::string imatrix_file;
    std::vector<std::string> included_weights, excluded_weights;
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<tensor_type_option> tensor_type_opts;
    std::vector<int> prune_layers;
    std::unordered_map<ggml_type, float> fudge_factors;

    // fork (ik_llama) options the merge dropped from this CLI; the fields they set are still read by src/llama-quantize.cpp
    std::vector<CustomQ> custom_quants;
    std::vector<std::string> repack_patterns;
    std::vector<uint16_t> only_splits; // --individual-tensors, 0-based
    quantize_user_data user_data = {};
    params.user_data = &user_data;
    bool hide_imatrix = false;

    // --cuda-quantize device selection (ik #2674/#2682). ik landed these options in the
    // examples/quantize copy, which this tree does not build; ported here so the shipped
    // llama-quantize can reach the encoder.
    std::vector<int> cuda_devs;
    std::vector<float> cuda_ts;
    bool have_cuda_ts = false;
    const std::pair<const char *, ggml_type *> fork_type_flags[] = {
        { "--extra-output-tensor",            &params.extra_output_type },
        { "--per-layer-token-embedding-type", &params.per_layer_token_embedding_type },
        { "--ffn-gate-inp-type",              &params.ffn_gate_inp_type },
        { "--attn-q-type",                    &params.attn_q_type },
        { "--attn-k-type",                    &params.attn_k_type },
        { "--attn-v-type",                    &params.attn_v_type },
        { "--attn-qkv-type",                  &params.attn_qkv_type },
        { "--attn-output-type",               &params.attn_output_type },
        { "--ffn-gate-type",                  &params.ffn_gate_type },
        { "--ffn-down-type",                  &params.ffn_down_type },
        { "--ffn-up-type",                    &params.ffn_up_type },
    };

    for (; arg_idx < argc && (strncmp(argv[arg_idx], "--", 2) == 0 || strcmp(argv[arg_idx], "-dev") == 0 || strcmp(argv[arg_idx], "-ts") == 0); arg_idx++) {
        ggml_type * fork_type_dst = nullptr;
        for (const auto & [flag, dst] : fork_type_flags) {
            if (strcmp(argv[arg_idx], flag) == 0) {
                fork_type_dst = dst;
            }
        }
        if (fork_type_dst) {
            if (arg_idx == argc-1 || (*fork_type_dst = parse_ggml_type(argv[++arg_idx])) == GGML_TYPE_COUNT) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--custom-q") == 0) {
            if (arg_idx == argc-1 || !parse_custom_quants(argv[++arg_idx], custom_quants)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--repack") == 0) {
            params.only_repack = true;
        } else if (strcmp(argv[arg_idx], "--repack-pattern") == 0) {
            if (arg_idx == argc-1) {
                usage(argv[0]);
            }
            const auto p = string_split<std::string>(argv[++arg_idx], ',');
            repack_patterns.insert(repack_patterns.end(), p.begin(), p.end());
        } else if (strcmp(argv[arg_idx], "--symmetric-q40") == 0) {
            user_data.symmetric_q4_0 = true;
        } else if (strcmp(argv[arg_idx], "--slow-iq2ks") == 0) {
            user_data.slow_iq2_ks = true;
        } else if (strcmp(argv[arg_idx], "--gptq-hessian") == 0) {
            if (arg_idx == argc-1) {
                usage(argv[0]);
            }
            static imatrix_hessian_file gptq_hessian;
            std::string err;
            if (!imatrix_hessian_load(argv[++arg_idx], gptq_hessian, &err)) {
                fprintf(stderr, "%s: --gptq-hessian: cannot load %s: %s\n", __func__, argv[arg_idx], err.c_str());
                return 1;
            }
            printf("%s: GPTQ Hessian %s: %zu matrices, block %d\n", __func__, argv[arg_idx], gptq_hessian.entries.size(), gptq_hessian.block_size);
            params.gptq_hessian_raw = &gptq_hessian;
            if (params.gptq_mode == 0) {
                params.gptq_mode = 1;
            }
        } else if (strcmp(argv[arg_idx], "--gptq-damp") == 0) {
            if (arg_idx == argc-1) {
                usage(argv[0]);
            }
            params.gptq_damp = std::stof(argv[++arg_idx]);
        } else if (strcmp(argv[arg_idx], "--gptq-no-propagate") == 0) {
            params.gptq_mode = 2;
        } else if (strcmp(argv[arg_idx], "--hide-imatrix") == 0) {
            hide_imatrix = true;
        } else if (strcmp(argv[arg_idx], "--ignore-imatrix-rules") == 0) {
            params.ignore_imatrix_rules = true;
        } else if (strcmp(argv[arg_idx], "--partial-requant") == 0) {
            params.partial_requant = true;
        } else if (strcmp(argv[arg_idx], "--leave-output-tensor") == 0) {
            params.quantize_output_tensor = false;
        } else if (strcmp(argv[arg_idx], "--output-tensor-type") == 0) {
            if (arg_idx < argc-1) {
                params.output_tensor_type = parse_ggml_type(argv[++arg_idx]);
                if (params.output_tensor_type == GGML_TYPE_COUNT) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--token-embedding-type") == 0) {
            if (arg_idx < argc-1) {
                params.token_embedding_type = parse_ggml_type(argv[++arg_idx]);
                if (params.token_embedding_type == GGML_TYPE_COUNT) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--tensor-type") == 0) {
            if (arg_idx == argc-1 || !parse_tensor_type(argv[++arg_idx], tensor_type_opts)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--tensor-type-file") == 0) {
            if (arg_idx == argc-1 || !parse_tensor_type_file(argv[++arg_idx], tensor_type_opts)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--fudge-factors") == 0) {
            if (arg_idx == argc-1 || !parse_fudge_factors(argv[++arg_idx], fudge_factors)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--prune-layers") == 0) {
            if (arg_idx == argc-1 || !parse_layer_prune(argv[++arg_idx], prune_layers)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--override-kv") == 0) {
            if (arg_idx == argc-1 || !string_parse_kv_override(argv[++arg_idx], kv_overrides)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--dry-run") == 0) {
            params.dry_run = true;
        } else if (strcmp(argv[arg_idx], "--slab-size") == 0) {
            // ik_llama's --slab-size (merged by a9fc5f64b into the unbuilt examples/quantize copy only). The library has
            // honoured llama_model_quantize_params::slab_size all along (src/llama-quantize.cpp, default 1 GiB); this
            // restores the user's way to set it. Value in MiB, 0 = never slab.
            if (arg_idx < argc-1) {
                try {
                    const std::string v = argv[++arg_idx];
                    size_t pos = 0;
                    const unsigned long long mib = std::stoull(v, &pos);
                    if (pos != v.size() || v[0] == '-') {
                        throw std::invalid_argument("trailing characters or a sign");
                    }
                    params.slab_size = (size_t) mib << 20;
                }
                catch (const std::exception & e) {
                    fprintf(stderr, "%s: invalid slab size '%s' (%s)\n", __func__, argv[arg_idx], e.what());
                    return 1;
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--cuda-quantize") == 0) {
            // KT CUDA encoder (Thireus d9e286846). The port first added this to the unbuilt examples/quantize copy,
            // where the built llama-quantize never saw it (measured: the CPU build printed usage, rc 1).
            params.cuda_quantize = true;
        } else if (strcmp(argv[arg_idx], "--allow-requantize") == 0) {
            params.allow_requantize = true;
        } else if (strcmp(argv[arg_idx], "--pure") == 0) {
            params.pure = true;
        } else if (strcmp(argv[arg_idx], "--imatrix") == 0) {
            if (arg_idx < argc-1) {
                imatrix_file = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--include-weights") == 0) {
            if (arg_idx < argc-1) {
                included_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--exclude-weights") == 0) {
            if (arg_idx < argc-1) {
                excluded_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-scale") == 0) {
            if (arg_idx == argc-1) {
                usage(argv[0]);
            }
            const char * m = argv[++arg_idx];
            if      (strcmp(m, "auto") == 0) { params.nvfp4_scale_mode = LLAMA_NVFP4_SCALE_AUTO; }
            else if (strcmp(m, "one")  == 0) { params.nvfp4_scale_mode = LLAMA_NVFP4_SCALE_ONE_LEVEL; }
            else if (strcmp(m, "two")  == 0) { params.nvfp4_scale_mode = LLAMA_NVFP4_SCALE_TWO_LEVEL; }
            else { usage(argv[0]); }
        } else if (strcmp(argv[arg_idx], "--keep-split") == 0) {
            params.keep_split = true;
        } else if (strcmp(argv[arg_idx], "--individual-tensors") == 0 || strcmp(argv[arg_idx], "--only-splits") == 0) {
            // 1-based shard numbers as in the file names (-0000N-of-0000M.gguf); stored 0-based
            if (arg_idx == argc-1) {
                usage(argv[0]);
            }
            for (const auto & item : string_split<std::string>(argv[++arg_idx], ',')) {
                if (item.empty()) {
                    continue;
                }
                char * end = nullptr;
                const long v = std::strtol(item.c_str(), &end, 10);
                if (end == item.c_str() || *end != '\0' || v < 1 || v > 65535) {
                    fprintf(stderr, "%s: invalid shard number '%s' (expected 1..65535)\n", __func__, item.c_str());
                    usage(argv[0]);
                }
                only_splits.push_back((uint16_t) (v - 1));
            }
            if (only_splits.empty()) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--skip-first-shard") == 0) {
            params.skip_first_shard = true;
        // --slab-size and --cuda-quantize: one parser each, above (both branches ported them; the canonical
        // --slab-size also rejects a sign or trailing characters, which std::stoull alone accepts)
        } else if (strcmp(argv[arg_idx], "--device") == 0 || strcmp(argv[arg_idx], "-dev") == 0) {
            if (arg_idx == argc-1) {
                usage(argv[0]);
            }
            std::vector<int> devs;
            if (!parse_cuda_device_list(argv[++arg_idx], devs)) {
                fprintf(stderr, "%s: invalid CUDA device '%s' (expected CUDA<N> list, e.g. CUDA0,CUDA1, or bare index)\n", __func__, argv[arg_idx]);
                return 1;
            }
            cuda_devs = std::move(devs);
        } else if (strcmp(argv[arg_idx], "--tensor-split") == 0 || strcmp(argv[arg_idx], "-ts") == 0) {
            if (arg_idx == argc-1) {
                usage(argv[0]);
            }
            std::vector<float> ts;
            for (const auto & item : string_split<std::string>(argv[++arg_idx], ',')) {
                try {
                    const float v = std::stof(item);
                    if (!std::isfinite(v) || v <= 0.0f) {
                        throw std::runtime_error("non-positive");
                    }
                    ts.push_back(v);
                }
                catch (const std::exception &) {
                    fprintf(stderr, "%s: invalid tensor-split entry '%s' in '%s' (expected positive numbers, e.g. -ts 3,3,2)\n", __func__, item.c_str(), argv[arg_idx]);
                    return 1;
                }
            }
            if (ts.empty()) {
                fprintf(stderr, "%s: empty tensor-split list '%s'\n", __func__, argv[arg_idx]);
                return 1;
            }
            cuda_ts = std::move(ts);
            have_cuda_ts = true;
        } else if (strcmp(argv[arg_idx], "--max-buffer-size") == 0) {
            if (arg_idx == argc-1) {
                usage(argv[0]);
            }
            const int mib = atoi(argv[++arg_idx]);
            if (mib <= 0) {
                fprintf(stderr, "%s: invalid --max-buffer-size '%s'\n", __func__, argv[arg_idx]);
                return 1;
            }
            params.max_buf_size = (size_t) mib * 1024 * 1024;
        } else {
            usage(argv[0]);
        }
    }

    // 1 device = mono, otherwise N-way shares (equal when -ts is absent).
    if (!cuda_devs.empty() || have_cuda_ts) {
        if (cuda_devs.empty()) {
            fprintf(stderr, "%s: -ts/--tensor-split needs -dev/--device with at least 2 CUDA devices\n", __func__);
            return 1;
        }
        if (have_cuda_ts && cuda_ts.size() != cuda_devs.size()) {
            fprintf(stderr, "%s: tensor-split entries (%zu) must match device count (%zu)\n", __func__, cuda_ts.size(), cuda_devs.size());
            return 1;
        }
        if (cuda_devs.size() > 16) {
            fprintf(stderr, "%s: at most 16 CUDA devices supported, got %zu\n", __func__, cuda_devs.size());
            return 1;
        }
        params.cuda_quantize_device = cuda_devs[0];
        if (cuda_devs.size() > 1) {
            params.cuda_quantize_n_devices = static_cast<int>(cuda_devs.size());
            float total = 0.0f;
            for (size_t i = 0; i < cuda_devs.size(); ++i) {
                total += have_cuda_ts ? cuda_ts[i] : 1.0f;
            }
            for (size_t i = 0; i < cuda_devs.size(); ++i) {
                params.cuda_quantize_devices[i] = cuda_devs[i];
                params.cuda_quantize_split[i] = (have_cuda_ts ? cuda_ts[i] : 1.0f)/total;
            }
        }
    }
    if (!only_splits.empty()) {
        if (!params.keep_split) {
            fprintf(stderr, "%s: --individual-tensors requires --keep-split\n", __func__);
            usage(argv[0]);
        }
        params.only_splits   = only_splits.data();
        params.n_only_splits = (int32_t) only_splits.size();
    } else if (params.skip_first_shard) {
        fprintf(stderr, "%s: --skip-first-shard requires --individual-tensors\n", __func__);
        usage(argv[0]);
    }

    if (argc - arg_idx < 2) {
        printf("%s: bad arguments\n", argv[0]);
        usage(argv[0]);
    }
    if (!included_weights.empty() && !excluded_weights.empty()) {
        usage(argv[0]);
    }
    for (const auto & [type, factor] : fudge_factors) {
        ggml_set_quantize_fudge_factor(type, factor);
    }
    if (!repack_patterns.empty()) {
        params.repack_pattern = &repack_patterns;
    }

    std::vector<std::string> imatrix_datasets;
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    int m_last_call = prepare_imatrix(imatrix_file, imatrix_datasets, included_weights, excluded_weights, imatrix_data);

    std::vector<llama_model_imatrix_data> i_data;
    std::vector<llama_model_tensor_override> t_override;
    if (!imatrix_data.empty()) {
        i_data.reserve(imatrix_data.size() + 1);
        for (const auto & kv : imatrix_data) {
            i_data.push_back({kv.first.c_str(), kv.second.data(), kv.second.size()});
        }
        i_data.push_back({nullptr, nullptr, 0});  // array terminator
        params.imatrix = i_data.data();
        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_FILE);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            strncpy(kvo.val_str, hide_imatrix ? "top_secret" : imatrix_file.c_str(), 127);
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }
        if (!imatrix_datasets.empty()) {
            llama_model_kv_override kvo;
            // TODO: list multiple datasets when there are more than one
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_DATASET);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            strncpy(kvo.val_str, hide_imatrix ? "top_secret" : imatrix_datasets[0].c_str(), 127);
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }
        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            kvo.val_i64 = hide_imatrix ? 0 : imatrix_data.size();
            kv_overrides.emplace_back(std::move(kvo));
        }
        if (m_last_call > 0) {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            kvo.val_i64 = hide_imatrix ? 0 : m_last_call;
            kv_overrides.emplace_back(std::move(kvo));
        }
    }
    if (!kv_overrides.empty()) {
        kv_overrides.emplace_back();
        kv_overrides.back().key[0] = 0;
        params.kv_overrides = kv_overrides.data();
    }
    if (!tensor_type_opts.empty()) {
        t_override.reserve(tensor_type_opts.size() + 1);
        for (const auto & tt : tensor_type_opts) {
            t_override.push_back({tt.name.c_str(), tt.type});
        }
        t_override.push_back({nullptr, GGML_TYPE_COUNT});  // array terminator
        params.tt_overrides = t_override.data();
    }
    if (!custom_quants.empty()) {
        params.custom_quants = &custom_quants;
    }
    if (!prune_layers.empty()) {
        prune_layers.push_back(-1);  // array terminator
        params.prune_layers = prune_layers.data();
    }

    llama_backend_init();

    // parse command line arguments
    const std::string fname_inp = argv[arg_idx];
    arg_idx++;
    std::string fname_out;

    std::string ftype_str;
    std::string suffix = ".gguf";
    if (try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
        // argv[arg_idx] is the ftype directly: <input> <ftype>
        if (!params.dry_run) {
            std::string fpath;
            const size_t pos = fname_inp.find_last_of("/\\");
            if (pos != std::string::npos) {
                fpath = fname_inp.substr(0, pos + 1);
            }

            // export as [inp path]/ggml-model-[ftype]. Only add extension if there is no splitting
            fname_out = fpath + "ggml-model-" + ftype_str;
            if (!params.keep_split) {
                fname_out += suffix;
            }
        }
        arg_idx++;
        if (ftype_str == "COPY") {
            params.only_copy = true;
        }
    } else {
        // argv[arg_idx] is not a valid ftype, so treat it as output path: <input> <output> <ftype>
        fname_out = argv[arg_idx];
        if (params.keep_split && fname_out.find(suffix) != std::string::npos) {
            fname_out = fname_out.substr(0, fname_out.length() - suffix.length());
        }
        arg_idx++;

        if (argc <= arg_idx) {
            fprintf(stderr, "%s: missing ftype\n", __func__);
            return 1;
        }
        if (!try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
            fprintf(stderr, "%s: invalid ftype '%s'\n", __func__, argv[arg_idx]);
            return 1;
        }
        if (ftype_str == "COPY") {
           params.only_copy = true;
        }
        arg_idx++;
    }

    // parse nthreads
    if (argc > arg_idx) {
        try {
            params.nthread = std::stoi(argv[arg_idx]);
        }
        catch (const std::exception & e) {
            fprintf(stderr, "%s: invalid nthread '%s' (%s)\n", __func__, argv[arg_idx], e.what());
            return 1;
        }
    }

    if (!params.dry_run) {
        if (std::error_code ec; std::filesystem::equivalent(fname_inp, fname_out, ec)) {
            fprintf(stderr, "%s: error: input and output files are the same: '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }
    }

    llama_print_build_info(llama_version());

    if (params.dry_run) {
        fprintf(stderr, "%s: calculating quantization size for '%s' as %s", __func__, fname_inp.c_str(), ftype_str.c_str());
    } else {
        fprintf(stderr, "%s: quantizing '%s' to '%s' as %s", __func__, fname_inp.c_str(), fname_out.c_str(), ftype_str.c_str());
    }

    if (params.nthread > 0) {
        fprintf(stderr, " using %d threads", params.nthread);
    }
    fprintf(stderr, "\n");

    const int64_t t_main_start_us = llama_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = llama_time_us();

        if (llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), &params)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = llama_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = llama_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    llama_backend_free();

    return 0;
}
