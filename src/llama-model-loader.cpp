#include "llama-model-loader.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-cpu.h"   // CPU-side declarations (ggml_graph_plan/compute, ggml_cpu_has_*, ggml_get/set_*) moved here, per upstream
#include "ggml-alloc.h"
#include <memory>
#include "ggml-backend.h"
#include "iqk/iqk_quantize.h"

#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#elif defined(GGML_USE_VULKAN)
#  include "ggml-vulkan.h"
#elif defined(GGML_USE_SYCL)
#  include "ggml-sycl.h"
#elif defined(GGML_USE_CANN)
#   include "ggml-cann.h"
#endif

#include <set>
#include <map>
#include <array>
#include <charconv>
#include <future>
#include <regex>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>
#include <cinttypes>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif
#include "llama-load-budget.h"   // after <windows.h>: the header includes it on _WIN32 as well

#define LLAMA_API_INTERNAL
#include <unordered_set>

static const size_t kiB = 1024;
static const size_t MiB = 1024*kiB;
static const size_t GiB = 1024*MiB;

// llama_file_version_name: fork's header declared this `static` (each TU gets its own copy,
// harmless -- src/llama.cpp includes the header and gets one too). Moved the definition here to
// match upstream's extern-declared-in-header / defined-once-in-.cpp shape, since this .cpp is the
// natural single home for it and the header no longer carries a body. Behavior identical.
const char * llama_file_version_name(llama_fver version) {
    switch (version) {
        case GGUF_FILE_VERSION_V1: return "GGUF V1 (support until nov 2023)";
        case GGUF_FILE_VERSION_V2: return "GGUF V2";
        case GGUF_FILE_VERSION_V3: return "GGUF V3 (latest)";
    }

    return "unknown";
}

// llama_ftype_name: REQUIRED, not optional -- it is declared LLAMA_API in include/llama.h and
// CONFIRMED called by tools/server/server-context.cpp (a forbidden-to-edit file read only to
// check this). Dropping it would leave a public symbol undefined at link time.
#define LLAMA_FTYPE_PREFIX "(guessed) "

const char * llama_ftype_name(llama_ftype ftype) {
    static constexpr size_t guessed_prefix_len = sizeof(LLAMA_FTYPE_PREFIX) - 1;
    const char * name;
    switch ((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) {
        case LLAMA_FTYPE_ALL_F32:          name = LLAMA_FTYPE_PREFIX "all F32"; break;
        case LLAMA_FTYPE_MOSTLY_F16:       name = LLAMA_FTYPE_PREFIX "F16"; break;
        case LLAMA_FTYPE_MOSTLY_BF16:      name = LLAMA_FTYPE_PREFIX "BF16"; break;
        case LLAMA_FTYPE_MOSTLY_Q1_0:      name = LLAMA_FTYPE_PREFIX "Q1_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_0:      name = LLAMA_FTYPE_PREFIX "Q2_0"; break;
        case LLAMA_FTYPE_MOSTLY_F8_E4M3:   name = LLAMA_FTYPE_PREFIX "F8_E4M3"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_0:      name = LLAMA_FTYPE_PREFIX "Q4_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1:      name = LLAMA_FTYPE_PREFIX "Q4_1"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0:      name = LLAMA_FTYPE_PREFIX "Q5_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1:      name = LLAMA_FTYPE_PREFIX "Q5_1"; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0:      name = LLAMA_FTYPE_PREFIX "Q8_0"; break;
        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: name = LLAMA_FTYPE_PREFIX "MXFP4 MoE"; break;
        case LLAMA_FTYPE_MOSTLY_NVFP4:     name = LLAMA_FTYPE_PREFIX "NVFP4"; break;
        case LLAMA_FTYPE_MOSTLY_NVFP4_KS:  name = LLAMA_FTYPE_PREFIX "NVFP4_KS"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_K:      name = LLAMA_FTYPE_PREFIX "Q2_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:    name = LLAMA_FTYPE_PREFIX "Q2_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:    name = LLAMA_FTYPE_PREFIX "Q3_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:    name = LLAMA_FTYPE_PREFIX "Q3_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:    name = LLAMA_FTYPE_PREFIX "Q3_K - Large"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:    name = LLAMA_FTYPE_PREFIX "Q4_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:    name = LLAMA_FTYPE_PREFIX "Q4_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:    name = LLAMA_FTYPE_PREFIX "Q5_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:    name = LLAMA_FTYPE_PREFIX "Q5_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:      name = LLAMA_FTYPE_PREFIX "Q6_K"; break;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:     name = LLAMA_FTYPE_PREFIX "TQ1_0 - 1.69 bpw ternary"; break;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:     name = LLAMA_FTYPE_PREFIX "TQ2_0 - 2.06 bpw ternary"; break;
        case LLAMA_FTYPE_MOSTLY_MXFP6_E2M3: name = LLAMA_FTYPE_PREFIX "MXFP6_E2M3 - 6.25 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_MXFP6_E3M2: name = LLAMA_FTYPE_PREFIX "MXFP6_E3M2 - 6.25 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_MXFP8_E4M3: name = LLAMA_FTYPE_PREFIX "MXFP8_E4M3 - 8.25 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_MXFP8_E5M2: name = LLAMA_FTYPE_PREFIX "MXFP8_E5M2 - 8.25 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_TQ3_1S:     name = LLAMA_FTYPE_PREFIX "TQ3_1S - 4.0 bpw WHT-rotated"; break;
        case LLAMA_FTYPE_MOSTLY_TQ4_1S:     name = LLAMA_FTYPE_PREFIX "TQ4_1S - 5.0 bpw WHT-rotated"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:   name = LLAMA_FTYPE_PREFIX "IQ2_XXS - 2.0625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:    name = LLAMA_FTYPE_PREFIX "IQ2_XS - 2.3125 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:     name = LLAMA_FTYPE_PREFIX "IQ2_S - 2.5 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:     name = LLAMA_FTYPE_PREFIX "IQ2_M - 2.7 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:    name = LLAMA_FTYPE_PREFIX "IQ3_XS - 3.3 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:   name = LLAMA_FTYPE_PREFIX "IQ3_XXS - 3.0625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:     name = LLAMA_FTYPE_PREFIX "IQ1_S - 1.5625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:     name = LLAMA_FTYPE_PREFIX "IQ1_M - 1.75 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_XS:    name = LLAMA_FTYPE_PREFIX "IQ1_XS - 1.4375 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_XXS:   name = LLAMA_FTYPE_PREFIX "IQ1_XXS - 1.3125 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_XXXS:  name = LLAMA_FTYPE_PREFIX "IQ1_XXXS - 1.1875 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:    name = LLAMA_FTYPE_PREFIX "IQ4_NL - 4.5 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:    name = LLAMA_FTYPE_PREFIX "IQ4_XS - 4.25 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:     name = LLAMA_FTYPE_PREFIX "IQ3_S - 3.4375 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:     name = LLAMA_FTYPE_PREFIX "IQ3_S mix - 3.66 bpw"; break;
        default:                           name = LLAMA_FTYPE_PREFIX "unknown, may not work"; break;
    }
    return (ftype & LLAMA_FTYPE_GUESSED) ? name : name + guessed_prefix_len;
}

#undef LLAMA_FTYPE_PREFIX

// return a list of splits for a given path
// for example, given "<name>-00002-of-00004.gguf", returns list of all 4 splits
static std::vector<std::string> llama_get_list_splits(const std::string & path, const int idx, const int n_split) {
    std::vector<std::string> paths;
    std::string split_prefix;
    std::vector<char> buf(llama_path_max(), 0);

    {
        int ret = llama_split_prefix(buf.data(), buf.size(), path.c_str(), idx, n_split);
        if (!ret) {
            throw std::runtime_error(format("invalid split file name: %s", path.c_str()));
        }
        split_prefix = std::string(buf.data(), ret);
    }

    if (split_prefix.empty()) {
        throw std::runtime_error(format("invalid split file: %s", path.c_str()));
    }

    for (int idx = 0; idx < n_split; ++idx) {
        int ret = llama_split_path(buf.data(), buf.size(), split_prefix.c_str(), idx, n_split);
        paths.push_back(std::string(buf.data(), ret));
    }

    return paths;
}

// GKV_Base_Type's gfun pointer type must be `(const gguf_context*, const int64_t)` to match
// gguf_get_val_* in ggml/include/gguf.h (already resolved by a sibling to upstream's int64_t
// key_id signature). Fork's original `const int` parameter does not bind to that function
// pointer type at all -- template argument deduction fails and the whole file does not compile.
// REQUIRED, not a style choice.
namespace GGUFMeta {
    template <typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int64_t)>
    struct GKV_Base_Type {
        static constexpr gguf_type gt = gt_;

        static T getter(const gguf_context * ctx, const int kid) {
            return gfun(ctx, kid);
        }
    };

    template<typename T> struct GKV_Base;

    template<> struct GKV_Base<bool        >: GKV_Base_Type<bool,         GGUF_TYPE_BOOL,    gguf_get_val_bool> {};
    template<> struct GKV_Base<uint8_t     >: GKV_Base_Type<uint8_t,      GGUF_TYPE_UINT8,   gguf_get_val_u8  > {};
    template<> struct GKV_Base<uint16_t    >: GKV_Base_Type<uint16_t,     GGUF_TYPE_UINT16,  gguf_get_val_u16 > {};
    template<> struct GKV_Base<uint32_t    >: GKV_Base_Type<uint32_t,     GGUF_TYPE_UINT32,  gguf_get_val_u32 > {};
    template<> struct GKV_Base<uint64_t    >: GKV_Base_Type<uint64_t,     GGUF_TYPE_UINT64,  gguf_get_val_u64 > {};
    template<> struct GKV_Base<int8_t      >: GKV_Base_Type<int8_t,       GGUF_TYPE_INT8,    gguf_get_val_i8  > {};
    template<> struct GKV_Base<int16_t     >: GKV_Base_Type<int16_t,      GGUF_TYPE_INT16,   gguf_get_val_i16 > {};
    template<> struct GKV_Base<int32_t     >: GKV_Base_Type<int32_t,      GGUF_TYPE_INT32,   gguf_get_val_i32 > {};
    template<> struct GKV_Base<int64_t     >: GKV_Base_Type<int64_t,      GGUF_TYPE_INT64,   gguf_get_val_i64 > {};
    template<> struct GKV_Base<float       >: GKV_Base_Type<float,        GGUF_TYPE_FLOAT32, gguf_get_val_f32 > {};
    template<> struct GKV_Base<double      >: GKV_Base_Type<double,       GGUF_TYPE_FLOAT64, gguf_get_val_f64 > {};
    template<> struct GKV_Base<const char *>: GKV_Base_Type<const char *, GGUF_TYPE_STRING,  gguf_get_val_str > {};

    template<> struct GKV_Base<std::string> {
        static constexpr gguf_type gt = GGUF_TYPE_STRING;

        static std::string getter(const gguf_context * ctx, const int kid) {
            return gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo {
        const gguf_type gt;
        const size_t length;
        const void * data;
    };

    template<> struct GKV_Base<ArrayInfo> {
        public:
        static constexpr gguf_type gt = GGUF_TYPE_ARRAY;
        static ArrayInfo getter(const gguf_context *ctx, const int k) {
            // upstream: gguf_get_arr_data() on a STRING array is undefined (strings aren't
            // packed as raw data); guard it. Fork's side called it unconditionally.
            const enum gguf_type arr_type = gguf_get_arr_type(ctx, k);
            return ArrayInfo {
                arr_type,
                size_t(gguf_get_arr_n(ctx, k)),
                arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx, k),
            };
        }
    };

    template<typename T>
    class GKV : public GKV_Base<T> {
        GKV() = delete;

        public:
        static T get_kv(const gguf_context * ctx, const int k) {
            const enum gguf_type kt = gguf_get_kv_type(ctx, k);

            if (kt != GKV::gt) {
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                    gguf_get_key(ctx, k), gguf_type_name(kt), gguf_type_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static const char * override_type_to_str(const llama_model_kv_override_type ty) {
            switch (ty) {
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:  return "bool";
                case LLAMA_KV_OVERRIDE_TYPE_INT:   return "int";
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT: return "float";
                case LLAMA_KV_OVERRIDE_TYPE_STR:   return "str";
            }
            return "unknown";
        }

        static bool validate_override(const llama_model_kv_override_type expected_type, const struct llama_model_kv_override * ovrd) {
            if (!ovrd) { return false; }
            if (ovrd->tag == expected_type) {
                LLAMA_LOG_INFO("%s: Using metadata override (%5s) '%s' = ",
                    __func__, override_type_to_str(ovrd->tag), ovrd->key);
                switch (ovrd->tag) {
                    case LLAMA_KV_OVERRIDE_TYPE_BOOL:  {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_bool ? "true" : "false");
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_INT:   {
                        LLAMA_LOG_INFO("%" PRId64 "\n", ovrd->val_i64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_FLOAT: {
                        LLAMA_LOG_INFO("%.6f\n", ovrd->val_f64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_STR: {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_str);
                    } break;
                    default:
                        // Shouldn't be possible to end up here, but just in case...
                        throw std::runtime_error(
                            format("Unsupported attempt to override %s type for metadata key %s\n",
                                override_type_to_str(ovrd->tag), ovrd->key));
                }
                return true;
            }
            LLAMA_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n",
                __func__, ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) {
                target = ovrd->val_bool;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
                target = ovrd->val_i64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) {
                target = ovrd->val_f64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_STR, ovrd)) {
                target = ovrd->val_str;
                return true;
            }
            return false;
        }

        static bool set(const gguf_context * ctx, const int k, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            if (try_override<T>(target, ovrd)) {
                return true;
            }
            if (k < 0) { return false; }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const gguf_context * ctx, const char * key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, gguf_find_key(ctx, key), target, ovrd);
        }

        static bool set(const gguf_context * ctx, const std::string & key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, key.c_str(), target, ovrd);
        }
    };
}

static bool parse_tensor_layer_index(const std::string & name, uint32_t & layer) {
    if (name.rfind("blk.", 0) != 0) {
        return false;
    }

    const char * first = name.data() + 4;
    const char * last  = first;
    const char * end   = name.data() + name.size();

    while (last < end && *last != '.') {
        ++last;
    }

    if (last == first || last == end) {
        return false;
    }

    auto result = std::from_chars(first, last, layer);
    return result.ec == std::errc() && result.ptr == last;
}

static bool is_split_expert_tensor(const std::string & name, uint32_t & expert) {
    static const char * prefixes[] = { "ffn_gate.", "ffn_down.", "ffn_up." };

    const size_t layer_end = name.find('.', 4);
    if (layer_end == std::string::npos) {
        return false;
    }

    const size_t prefix_begin = layer_end + 1;

    for (const char * prefix : prefixes) {
        const size_t prefix_len = std::char_traits<char>::length(prefix);
        if (name.compare(prefix_begin, prefix_len, prefix) != 0) {
            continue;
        }

        const size_t expert_begin = prefix_begin + prefix_len;
        const size_t expert_end = name.find('.', expert_begin);
        if (expert_end == std::string::npos || expert_end == expert_begin) {
            continue;
        }

        auto result = std::from_chars(name.data() + expert_begin, name.data() + expert_end, expert);
        if (result.ec == std::errc() && result.ptr == name.data() + expert_end) {
            return true;
        }
    }

    return false;
}

static bool is_merged_expert_tensor(llm_tensor tensor_type) {
    switch (tensor_type) {
        case LLM_TENSOR_FFN_NORM_EXPS:
        case LLM_TENSOR_FFN_DOWN_EXPS:
        case LLM_TENSOR_FFN_GATE_EXPS:
        case LLM_TENSOR_FFN_UP_EXPS:
        case LLM_TENSOR_FFN_GATE_UP_EXPS:
        case LLM_TENSOR_FFN_EXP_PROBS_B:
            return true;
        default:
            return false;
    }
}

static void coalesce_ranges(std::vector<llama_file_range> & ranges) {
    ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [](const llama_file_range & range) {
        return range.empty();
    }), ranges.end());

    std::sort(ranges.begin(), ranges.end(), [](const llama_file_range & lhs, const llama_file_range & rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return lhs.last < rhs.last;
    });

    std::vector<llama_file_range> merged;
    merged.reserve(ranges.size());

    for (const auto & range : ranges) {
        if (merged.empty() || range.first > merged.back().last) {
            merged.push_back(range);
            continue;
        }
        merged.back().last = std::max(merged.back().last, range.last);
    }

    ranges = std::move(merged);
}

// A GGUF's keys are prefixed with ITS OWN general.architecture string. When that string resolved to one of
// our archs through an alias (spelling alias or an Ollama name, see llm_arch_from_string), read keys under the
// file's string, not our canonical one -- otherwise every "<arch>.*" key is "not found". For a file whose
// name IS our canonical spelling this sets nothing. `file_arch` must outlive `kv` (it is the loader's member).
static void llm_kv_use_file_prefix(LLM_KV & kv, const std::string & file_arch) {
    if (kv.arch == LLM_ARCH_UNKNOWN || file_arch.empty()) {
        return;
    }
    const std::string canon = LLM_KV(kv.arch)(LLM_KV_BLOCK_COUNT);   // "<canonical>.block_count"
    if (canon.compare(0, file_arch.size() + 1, file_arch + ".") != 0) {
        kv.prefix = file_arch.c_str();
        LLAMA_LOG_INFO("%s: architecture '%s' is an alias of '%s'; reading its keys as '%s.*'\n", __func__,
                       file_arch.c_str(), canon.substr(0, canon.size() - strlen(".block_count")).c_str(), file_arch.c_str());
    }
}

// Ollama-format embeddinggemma: general.architecture=gemma3 plus the SentenceTransformers projection stored as dense.0 / dense.1
// (only embeddinggemma ships them). llama.cpp loads that model as gemma-embedding, whose dense_2 / dense_3 are looked up by name with
// shapes taken from <arch>.dense_{2,3}_feat_{in,out}. Detection, renames and the shape-derived KVs follow
// llama/compat/llama-ollama-compat.cpp:385-428. Keys stay under the file's own "gemma3." prefix (llm_kv_use_file_prefix), so nothing is
// copied; the three tensors are only re-indexed under the names the gemma-embedding loader asks for.
void llama_model_loader::apply_ollama_compat() {
    if (arch_name != "gemma3") {
        return;
    }
    auto d0 = weights_map.find("dense.0.weight");
    auto d1 = weights_map.find("dense.1.weight");
    if (d0 == weights_map.end() || d1 == weights_map.end()) {
        return;
    }
    const ggml_tensor * t0 = d0->second.tensor;
    const ggml_tensor * t1 = d1->second.tensor;
    gguf_set_val_u32(meta, "gemma3.dense_2_feat_in",  (uint32_t) t0->ne[0]);
    gguf_set_val_u32(meta, "gemma3.dense_2_feat_out", (uint32_t) t0->ne[1]);
    gguf_set_val_u32(meta, "gemma3.dense_3_feat_in",  (uint32_t) t1->ne[0]);
    gguf_set_val_u32(meta, "gemma3.dense_3_feat_out", (uint32_t) t1->ne[1]);
    if (gguf_find_key(meta, "gemma3.rope.freq_base_swa") < 0) {
        gguf_set_val_f32(meta, "gemma3.rope.freq_base_swa", 10000.0f);
    }
    auto rename = [&](const char * from, const char * to) {
        auto it = weights_map.find(from);
        if (it == weights_map.end() || weights_map.count(to)) {
            return;
        }
        llama_tensor_weight w = it->second;
        weights_map.erase(it);
        // `weights` (walked by index in get_layer_sizes / ftype detection) shares this ggml tensor, so its name changes too
        ggml_set_name(w.tensor, to);
        weights_map.emplace(to, w);
    };
    rename("dense.0.weight", "dense_2.weight");
    rename("dense.1.weight", "dense_3.weight");
    rename("norm.weight",    "output_norm.weight");
    llm_kv = LLM_KV(LLM_ARCH_GEMMA_EMBEDDING);
    llm_kv_use_file_prefix(llm_kv, arch_name);
    LLAMA_LOG_INFO("%s: detected Ollama-format embeddinggemma (gemma3 + dense.0/dense.1); loading as gemma-embedding\n", __func__);
}

// Ollama-format qwen35 (dense) with the MTP head embedded as mtp.*: llama.cpp's converter (conversion/qwen.py filter_tensors)
// puts that head at blk.<n_main>.* plus blk.<n_main>.nextn.{eh_proj,enorm,hnorm,shared_head_norm}, adds +1.0 to every
// *norm.weight (qwen.py modify_tensors), raises block_count by one and writes nextn_predict_layers. Ollama's blob keeps the
// head under mtp.* with block_count = n_main, so the fork loaded the trunk and left the head unused (done_getting_tensors).
//
// SOURCE-READ: tmp/arex1014/dc1017/ol_compat.cpp:583-895 (git show a8aaf9f:llama/compat/llama-ollama-compat.cpp,
// ollama, MIT). Design taken: rename, raise block_count, set nextn_predict_layers, +1.0 on norms at read time. (ollama also turns
// mmap off for that load; here only the shifted norms get owned storage, see load_all_data.) Written for this loader; no code copied. GATE-CLAIM-OK: design note, no resource is claimed unavailable.
// SOURCE-NUANCE: ollama's rule (qwen35_should_shift_norm_after_rename, ol_compat.cpp:619-624) shifts EVERY renamed tensor ending
// in _norm.weight, so the four norms inside mtp.layers.0 get +1.0 too. Measured 2026-10-17 against the HF source, Qwen/Qwen3.6-27B
// (BF16, HTTP range reads of model.safetensors) vs the ollama qwen3.6:27b blob (sha256-83c54730...):
//   mtp.layers.0.input_layernorm      HF mean 0.0406  blob mtp.layers.0.attn_norm            mean 1.0406  (already +1)
//   mtp.layers.0.post_attention_ln    HF mean 0.2108  blob mtp.layers.0.post_attention_norm  mean 1.2108  (already +1)
//   mtp.layers.0.self_attn.q/k_norm   HF mean 0.755 / 0.743   blob attn_q/k_norm  1.755 / 1.743       (already +1)
//   mtp.norm                          HF mean 1.2740  blob mtp.norm                          mean 1.2740  (raw)
//   mtp.pre_fc_norm_embedding/hidden  HF mean -0.4400 / -0.1711   blob identical                     (raw)
// The blob's converter already shifted the tensors it renamed to GGUF names and left the three HF-named ones raw, so shifting
// the four layer norms again would double them. Only the three raw ones are shifted here.
bool llama_model_loader::apply_ollama_qwen35_mtp_compat() {
    if (arch_name != "qwen35") {
        return false;
    }
    if (const char * e = getenv("LLAMA_OLLAMA_COMPAT"); e && std::string(e) == "0") {
        return false;
    }
    bool has_mtp = false;
    for (const auto & it : weights_map) {
        const std::string & n = it.first;
        if (n.compare(0, 4, "blk.") == 0 && n.find(".nextn.") != std::string::npos) {
            return false;   // native llama.cpp layout, nothing to translate
        }
        if (n.compare(0, 4, "mtp.") == 0) {
            has_mtp = true;
            if (n.compare(0, 13, "mtp.layers.0.") != 0 && n.compare(0, 11, "mtp.layers.") == 0) {
                LLAMA_LOG_WARN("%s: Ollama qwen35 blob has more than one MTP layer (%s); not translated\n", __func__, n.c_str());
                return false;
            }
            if (n.find("exps") != std::string::npos || n.find("experts") != std::string::npos) {
                LLAMA_LOG_WARN("%s: Ollama qwen35 MTP head has expert tensors (%s); not translated\n", __func__, n.c_str());
                return false;
            }
        }
    }
    if (!has_mtp) {
        return false;
    }
    static const char * required[] = {
        "mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.layers.0.attn_norm.weight",
    };
    for (const char * r : required) {
        if (!weights_map.count(r)) {
            LLAMA_LOG_WARN("%s: Ollama qwen35 MTP head lacks %s; not translated\n", __func__, r);
            return false;
        }
    }

    uint32_t n_main = 0;
    get_key(llm_kv(LLM_KV_BLOCK_COUNT), n_main);
    const uint32_t n_new = n_main + 1;

    // per-layer arrays written with n_main entries (Ollama: attention.head_count_kv = [0,0,0,4,...]) get one more entry for
    // the MTP layer, which is a full-attention layer: the largest value in the array
    for (const enum llm_kv kid : { LLM_KV_ATTENTION_HEAD_COUNT, LLM_KV_ATTENTION_HEAD_COUNT_KV }) {
        const std::string key = llm_kv(kid);
        const int64_t id = gguf_find_key(meta, key.c_str());
        if (id < 0 || gguf_get_kv_type(meta, id) != GGUF_TYPE_ARRAY) {
            continue;
        }
        const enum gguf_type at = gguf_get_arr_type(meta, id);
        const size_t n = gguf_get_arr_n(meta, id);
        if (n != n_main || (at != GGUF_TYPE_UINT32 && at != GGUF_TYPE_INT32)) {
            LLAMA_LOG_WARN("%s: %s has %zu entries of type %d (block_count %u); not translated\n", __func__, key.c_str(), n, (int) at, n_main);
            return false;
        }
        std::vector<uint32_t> v(n_new);
        memcpy(v.data(), gguf_get_arr_data(meta, id), n * sizeof(uint32_t));
        uint32_t mx = 0;
        for (size_t i = 0; i < n; ++i) {
            mx = std::max(mx, v[i]);
        }
        v[n] = mx;
        gguf_set_arr_data(meta, key.c_str(), at, v.data(), v.size());
    }
    gguf_set_val_u32(meta, llm_kv(LLM_KV_BLOCK_COUNT).c_str(), n_new);
    gguf_set_val_u32(meta, llm_kv(LLM_KV_NEXTN_PREDICT_LAYERS).c_str(), 1);

    auto rename = [&](const std::string & from, const std::string & to) {
        auto it = weights_map.find(from);
        if (it == weights_map.end() || weights_map.count(to)) {
            return false;
        }
        llama_tensor_weight w = it->second;
        weights_map.erase(it);
        ggml_set_name(w.tensor, to.c_str());   // `weights` shares this tensor, so its name follows
        weights_map.emplace(to, w);
        return true;
    };
    const std::string blk = "blk." + std::to_string(n_main) + ".";
    std::vector<std::string> layer_names;
    for (const auto & it : weights_map) {
        if (it.first.compare(0, 13, "mtp.layers.0.") == 0) {
            layer_names.push_back(it.first);
        }
    }
    int n_renamed = 0;
    for (const std::string & from : layer_names) {
        n_renamed += rename(from, blk + from.substr(13));
    }
    n_renamed += rename("mtp.fc.weight", blk + "nextn.eh_proj.weight");
    struct { const char * from; const char * to; } shifted[] = {
        { "mtp.pre_fc_norm_embedding.weight", "nextn.enorm.weight"            },
        { "mtp.pre_fc_norm_hidden.weight",    "nextn.hnorm.weight"            },
        { "mtp.norm.weight",                  "nextn.shared_head_norm.weight" },
    };
    for (const auto & s : shifted) {
        auto it = weights_map.find(s.from);
        if (it == weights_map.end()) {
            continue;
        }
        if (it->second.tensor->type != GGML_TYPE_F32) {
            throw std::runtime_error(format("Ollama qwen35 MTP compat: %s is %s; the +1.0 shift is written for F32 only",
                                            s.from, ggml_type_name(it->second.tensor->type)));
        }
        if (rename(s.from, blk + s.to)) {
            n_renamed++;
            compat_norm_add_one.push_back(blk + s.to);
        }
    }
    LLAMA_LOG_INFO("%s: Ollama qwen35 MTP head translated: %d mtp.* tensors -> blk.%u, block_count %u -> %u, nextn_predict_layers 1, "
                   "+1.0 on %zu norms at read time (owned storage for those, mmap unchanged)\n", __func__, n_renamed, n_main, n_main, n_new,
                   compat_norm_add_one.size());
    return true;
}

llama_model_loader::llama_model_loader(const std::string & fname, int ncmoe, bool use_mmap, bool check_tensors,
        bool repack_tensors, bool use_thp, bool merge_qkv, bool merge_up_gate_exps, bool defer_experts,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p,
        const std::vector<std::string> & splits,
        const std::vector<uint16_t> * only_splits,
        int nvfp4_precision_in) {
    nvfp4_precision = nvfp4_precision_in;
    int trace = 0;
    if (getenv("LLAMA_TRACE")) {
        trace = atoi(getenv("LLAMA_TRACE"));
    }

#ifdef _WIN32
    // Only bump maxstdio if the user really wants large contexts:
#if defined(GGML_MAX_CONTEXTS) && (GGML_MAX_CONTEXTS > 512)
    // Cap at MSVC's hard limit of 8192 - https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/setmaxstdio?view=msvc-160
#if (GGML_MAX_CONTEXTS > 8192)
#define _GGML_STDIO_TARGET 8192
#else
#define _GGML_STDIO_TARGET GGML_MAX_CONTEXTS
#endif
    int _setmaxstdio_ret = _setmaxstdio(_GGML_STDIO_TARGET);
    if (_setmaxstdio_ret == -1) {
        LLAMA_LOG_INFO("%s: failed to set max stdio to %d. (setmaxstdio returned -1)\n", __func__, _GGML_STDIO_TARGET);
    } else {
        LLAMA_LOG_INFO("%s: max stdio successfully set to %d\n", __func__, _setmaxstdio_ret);
    }
#endif // GGML_MAX_CONTEXTS > 512
#endif // _WIN32

    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            kv_overrides.insert({std::string(p->key), *p});
        }
    }

    tensor_buft_overrides = param_tensor_buft_overrides_p;

    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    meta = gguf_init_from_file(fname.c_str(), params);
    if (!meta) {
        throw std::runtime_error(format("%s: failed to load model from %s\n", __func__, fname.c_str()));
    }
    // `metadata` is upstream's name for this SAME context -- CONFIRMED needed by
    // src/llama-quant.cpp's `gguf_set_kv(ctx_out.get(), ml.metadata)` (already resolved, real
    // consumer). Aliased, not separately owned: the destructor frees only `meta`.
    metadata = meta;

    get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
    llm_kv = LLM_KV(llm_arch_from_string(arch_name));
    llm_kv_use_file_prefix(llm_kv, arch_name);

    files.emplace_back(new llama_file(fname.c_str(), "rb"));
    file_split_no.push_back(0);
    contexts.emplace_back(ctx);

    // Save tensors data offset of the main file.
    // For subsidiary files, `meta` tensor data offset must not be used,
    // so we build a unified tensors index for weights.
    // weights_map is populated in lock-step with `weights` from the SAME llama_tensor_weight
    // value (copy is cheap: idx/offs/tensor-pointer only) so the two containers can never
    // disagree -- weights_map is upstream's by-name index, CONFIRMED needed by
    // src/llama-quant.cpp (see the loader.h field comment for the full citation).
    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        weights.emplace_back(files.back().get(), 0, cur->name, meta, cur);
        weights_map.emplace(cur->name, weights.back());
    }
    apply_ollama_compat();
    uint16_t n_split = 0;
    get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

    // Load additional GGML contexts
    if (n_split > 1) {
        uint16_t idx = 0;
        get_key(llm_kv(LLM_KV_SPLIT_NO), idx);
        if (idx != 0) {
            throw std::runtime_error(format("illegal split file: %d, model must be loaded with the first split", idx));
        }

        // Resolve the split set. An EXPLICIT list from the caller (llama_model_load_from_splits)
        // wins; otherwise fall back to deriving it from the <name>-0000i-of-0000n.gguf naming
        // scheme via llama_get_list_splits(), which was already present in this file but had no
        // caller. The derived branch throws the same two runtime_errors, on the same conditions,
        // as the inline llama_split_prefix() code it replaces -- behaviour for every existing
        // call site (all of which pass no splits) is unchanged.
        std::vector<std::string> split_paths = splits;
        if (split_paths.empty()) {
            split_paths = llama_get_list_splits(fname, idx, n_split);
        } else if ((int) split_paths.size() != (int) n_split) {
            throw std::runtime_error(format(
                        "%s: %d splits were supplied but the model declares %d",
                        __func__, (int) split_paths.size(), (int) n_split));
        }

        if (trace > 0) {
            LLAMA_LOG_INFO("%s: loading additional %d GGUFs\n", __func__, n_split);
        }

        if (only_splits) {
            for (uint16_t s : *only_splits) {
                if (s >= n_split) {
                    throw std::runtime_error(format("%s: requested split %d but the model has %d splits",
                                __func__, (int) s + 1, (int) n_split));
                }
            }
        }

        int n_opened = 0;
        for (idx = 1; idx < n_split; idx++) {
            if (only_splits && std::find(only_splits->begin(), only_splits->end(), idx) == only_splits->end()) {
                partial_splits = true;
                continue; // not requested: the file is never opened and need not exist
            }
            const char * split_path = split_paths[idx].c_str();

            struct gguf_init_params split_params = {
                /*.no_alloc = */ true,
                /*.ctx      = */ &ctx,
            };
            struct gguf_context * ctx_gguf = gguf_init_from_file(split_path, split_params);
            if (!ctx_gguf) {
                throw std::runtime_error(format("%s: failed to load GGUF split from %s\n", __func__, split_path));
            }

            // weights carry the index into files (== idx when every split is opened)
            const uint16_t file_idx = (uint16_t) files.size();
            files.emplace_back(new llama_file(split_path, "rb"));
            file_split_no.push_back(idx);
            contexts.emplace_back(ctx);

            // Save tensors data offset info of the shard.
            for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
                weights.emplace_back(files.back().get(), file_idx, cur->name, ctx_gguf, cur);
                weights_map.emplace(cur->name, weights.back());
            }

            gguf_free(ctx_gguf);
            ++n_opened;
        }

        get_key(llm_kv(LLM_KV_SPLIT_TENSORS_COUNT), n_tensors);

        // sanity check
        {
            const int n_tensors_loaded = (int) weights.size();
            if (partial_splits) {
                if (n_tensors_loaded > n_tensors) {
                    throw std::runtime_error(format("corrupted model: at most %d tensors expected but %d found", n_tensors, n_tensors_loaded));
                }
                LLAMA_LOG_INFO("%s: partial split set: %d of %d additional GGUFs opened, %d of %d tensors\n",
                        __func__, n_opened, n_split - 1, n_tensors_loaded, n_tensors);
            } else if (n_tensors != n_tensors_loaded) {
                throw std::runtime_error(format("corrupted model: %d tensors expected but %d found", n_tensors, n_tensors_loaded));
            }
        }

        LLAMA_LOG_INFO("%s: additional %d GGUFs metadata loaded.\n",  __func__, n_opened);
    }

    // ik 30903876b: mainline NVFP4 (type 40) tensors -> NVFP4_KS at load. OPT-IN here (LLAMA_NVFP4_IMPORT_KS=1): the fork
    // runs type 40 natively (iqk AVX2 kernel, CUDA mmvq, W4A4 prefill, graph-side scale), ik has no native type 40.
    if (import_nvfp4()) {
        use_mmap = false;
    }

    n_kv      = gguf_get_n_kv(meta);
    n_tensors = weights.size();

    fver = (enum llama_fver) gguf_get_version(meta);

    std::set<std::string> tensor_names;
    for (auto & w : weights) {
        n_elements += ggml_nelements(w.tensor);
        n_bytes    += ggml_nbytes(w.tensor);
        // make sure there is no duplicated tensor names
        const std::string name(w.tensor->name);
        auto found = tensor_names.find(name);
        if (found != tensor_names.end()) {
            throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", w.tensor->name));
        }
        tensor_names.insert(name);
    }

    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
            __func__, n_kv, n_tensors, fname.c_str(), llama_file_version_name(fver));

    // determine file type based on the number of tensors for each quantization and print meta data
    // TODO: make optional
    {
        std::map<enum ggml_type, uint32_t> n_type;

        uint32_t n_type_max = 0;
        enum ggml_type type_max = GGML_TYPE_F32;

        for (int i = 0; i < n_tensors; i++) {
            const ggml_tensor * tensor = weights.at(i).tensor;
            enum ggml_type type = tensor->type;

            // NVFP4 per-tensor scale companions ("<name>.scale", one F32 each) would otherwise outvote the real weights
            {
                const char * nm = ggml_get_name(tensor);
                const size_t ln = strlen(nm);
                if (type == GGML_TYPE_F32 && ln > 6 && strcmp(nm + ln - 6, ".scale") == 0 && ggml_nelements(tensor) == 1) {
                    continue;
                }
            }

            n_type[type]++;

            if (n_type_max < n_type[type]) {
                n_type_max = n_type[type];
                type_max   = type;
            }

            if (trace > 0) {
                const uint16_t sid = weights.at(i).idx;
                // MiB suffix adopted from upstream -- strictly more information, no format lost.
                LLAMA_LOG_INFO("%s: - tensor %4d, split %2d: %32s %-8s [ %s ] %8.2f MiB\n", __func__, i, sid,
                        ggml_get_name(tensor), ggml_type_name(type), llama_format_tensor_shape(tensor).c_str(),
                        ggml_nbytes(tensor)/1024.0f/1024.0f);
            }
        }

        switch (type_max) {
            // Bonsai types. Without these the switch falls through to the unknown default and the
            // label comes from the file's general.file_type KV, which for these files is off by
            // one against our enum -- so a 2.126 bpw model printed as "Q1_0_G128 - 1.125 bpw".
            // type_max is the most-frequent tensor type (498 quantized vs 353 F32 in both files),
            // so it identifies the format from what the file actually CONTAINS.
            case GGML_TYPE_Q1_0_G128:ftype = LLAMA_FTYPE_MOSTLY_Q1_0_G128;break;
            case GGML_TYPE_Q2_0_G128:ftype = LLAMA_FTYPE_MOSTLY_Q2_0_G128;break;
            case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
            case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
            case GGML_TYPE_BF16:    ftype = LLAMA_FTYPE_MOSTLY_BF16;    break;
            case GGML_TYPE_TQ1_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ1_0;   break;
            case GGML_TYPE_TQ2_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ2_0;   break;
            case GGML_TYPE_BF16_R16:ftype = LLAMA_FTYPE_MOSTLY_BF16_R16;break;
            case GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
            case GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
            case GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
            case GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
            case GGML_TYPE_Q6_0:    ftype = LLAMA_FTYPE_MOSTLY_Q6_0;    break;
            case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
            case GGML_TYPE_Q8_KV:   ftype = LLAMA_FTYPE_MOSTLY_Q8_KV;   break;
            case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
            case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
            case GGML_TYPE_Q3_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q3_K_R4; break;
            case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
            case GGML_TYPE_Q4_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q4_K_R4; break;
            case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
            case GGML_TYPE_Q5_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q5_K_R4; break;
            case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
            case GGML_TYPE_Q6_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q6_K_R4; break;
            case GGML_TYPE_Q8_K_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_K_R8; break;
            case GGML_TYPE_Q8_K_R16:ftype = LLAMA_FTYPE_MOSTLY_Q8_K_R16;break;
            case GGML_TYPE_Q8_KV_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_KV_R8; break;
            case GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
            case GGML_TYPE_IQ2_XXS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4; break;
            case GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
            case GGML_TYPE_IQ2_XS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS_R4; break;
            case GGML_TYPE_IQ2_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KS;  break;
            case GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_M;   break;
            case GGML_TYPE_IQ2_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_M_R4;break;
            case GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
            case GGML_TYPE_IQ3_XXS_R4: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4; break;
            case GGML_TYPE_IQ1_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ1_KT;  break;
            case GGML_TYPE_PQ2_0:
            case GGML_TYPE_PQ2_0_R8: ftype = LLAMA_FTYPE_MOSTLY_PQ2_0; break;
            case GGML_TYPE_PTQ1_0:
            case GGML_TYPE_PTQ1_0_R8:ftype = LLAMA_FTYPE_MOSTLY_PTQ1_0;break;
            case GGML_TYPE_TQK6:     ftype = LLAMA_FTYPE_MOSTLY_TQK6;  break;
            case GGML_TYPE_TQK7:     ftype = LLAMA_FTYPE_MOSTLY_TQK7;  break;
            case GGML_TYPE_TQ2_T:    ftype = LLAMA_FTYPE_MOSTLY_TQ2_T; break;
            case GGML_TYPE_TQP5:     ftype = LLAMA_FTYPE_MOSTLY_TQP5;  break;
            case GGML_TYPE_TQP6:     ftype = LLAMA_FTYPE_MOSTLY_TQP6;  break;
            case GGML_TYPE_TQP7:     ftype = LLAMA_FTYPE_MOSTLY_TQP7;  break;
            case GGML_TYPE_IQ2_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KT;  break;
            case GGML_TYPE_IQ3_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ3_KT;  break;
            case GGML_TYPE_IQ4_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_KT;  break;
            case GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
            case GGML_TYPE_IQ1_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ1_S_R4;break;
            case GGML_TYPE_IQ1_M_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ1_M_R4;break;
            case GGML_TYPE_IQ1_M:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;   break;
            case GGML_TYPE_IQ1_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ1_XS;  break;
            case GGML_TYPE_IQ1_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ1_XXS; break;
            case GGML_TYPE_IQ1_XXXS:ftype = LLAMA_FTYPE_MOSTLY_IQ1_XXXS;break;
            case GGML_TYPE_IQ1_BN:  ftype = LLAMA_FTYPE_MOSTLY_IQ1_BN;  break;
            case GGML_TYPE_IQ2_BN:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_BN;  break;
            case GGML_TYPE_IQ2_BN_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_BN_R4;break;
            case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
            case GGML_TYPE_IQ4_NL_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL_R4;break;
            case GGML_TYPE_IQ4_XS_R8:ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS_R8;break;
            case GGML_TYPE_Q4_0_R8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_R8; break;
            case GGML_TYPE_Q2_0_R8: ftype = LLAMA_FTYPE_MOSTLY_Q2_0_R8; break;
            case GGML_TYPE_Q5_0_R4: ftype = LLAMA_FTYPE_MOSTLY_Q5_0_R4; break;
            case GGML_TYPE_Q6_0_R4: ftype = LLAMA_FTYPE_MOSTLY_Q6_0_R4; break;
            case GGML_TYPE_Q8_0_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_0_R8; break;
            case GGML_TYPE_MXFP4:   ftype = LLAMA_FTYPE_MOSTLY_MXFP4;   break;
            case GGML_TYPE_MXFP6_E2M3: ftype = LLAMA_FTYPE_MOSTLY_MXFP6_E2M3; break;
            case GGML_TYPE_MXFP6_E3M2: ftype = LLAMA_FTYPE_MOSTLY_MXFP6_E3M2; break;
            case GGML_TYPE_MXFP8_E4M3: ftype = LLAMA_FTYPE_MOSTLY_MXFP8_E4M3; break;
            case GGML_TYPE_MXFP8_E5M2: ftype = LLAMA_FTYPE_MOSTLY_MXFP8_E5M2; break;
            case GGML_TYPE_TQ3_1S:     ftype = LLAMA_FTYPE_MOSTLY_TQ3_1S;     break;
            case GGML_TYPE_TQ4_1S:     ftype = LLAMA_FTYPE_MOSTLY_TQ4_1S;     break;
            case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
            case GGML_TYPE_IQ4_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_KS;  break;
            case GGML_TYPE_IQ4_KS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_KS_R4;  break;
            case GGML_TYPE_IQ5_KS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ5_KS_R4;  break;
            case GGML_TYPE_IQ3_KS_R16:ftype = LLAMA_FTYPE_MOSTLY_IQ3_KS_R16;break;
            case GGML_TYPE_IQ4_KS_R16:ftype = LLAMA_FTYPE_MOSTLY_IQ4_KS_R16;break;
            case GGML_TYPE_IQ5_KS_R16:ftype = LLAMA_FTYPE_MOSTLY_IQ5_KS_R16;break;
            case GGML_TYPE_IQ4_KSS: ftype = LLAMA_FTYPE_MOSTLY_IQ4_KSS; break;
            case GGML_TYPE_IQ5_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ5_KS;  break;
            case GGML_TYPE_IQ2_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_K;   break;
            case GGML_TYPE_IQ2_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_K_R4;break;
            case GGML_TYPE_IQ3_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ3_KS;  break;
            case GGML_TYPE_IQ2_KL:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KL;  break;
            case GGML_TYPE_IQ3_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_K;   break;
            case GGML_TYPE_IQ3_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ3_K_R4;break;
            case GGML_TYPE_IQ4_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ4_K;   break;
            case GGML_TYPE_IQ4_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_K_R4;break;
            case GGML_TYPE_IQ5_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ5_K;   break;
            case GGML_TYPE_IQ5_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ5_K_R4;break;
            case GGML_TYPE_IQ6_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ6_K;   break;
            case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
            case GGML_TYPE_IQ3_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ3_S_R4;break;
            case GGML_TYPE_Q4_0_4_4: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_4_4; break;
            case GGML_TYPE_Q4_0_4_8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_4_8; break;
            case GGML_TYPE_Q4_0_8_8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_8_8; break;
            // adopted from upstream in this merge -- these 3 ggml types (ggml.h) and their
            // corresponding llama_ftype enumerators (include/llama.h) were added in the SAME
            // 2026-09-04 upstream-ftype-merge; without these cases a NVFP4/Q1_0/Q2_0 quantized
            // file's dominant tensor type would fall through to the "unknown" default below.
            case GGML_TYPE_NVFP4:   ftype = LLAMA_FTYPE_MOSTLY_NVFP4;   break;
            case GGML_TYPE_NVFP4_KS:ftype = LLAMA_FTYPE_MOSTLY_NVFP4_KS;break;
            case GGML_TYPE_Q1_0:    ftype = LLAMA_FTYPE_MOSTLY_Q1_0;    break;
            case GGML_TYPE_Q2_0:    ftype = LLAMA_FTYPE_MOSTLY_Q2_0;    break;
            case GGML_TYPE_F8_E4M3: ftype = LLAMA_FTYPE_MOSTLY_F8_E4M3; break;
            default:
                {
                     LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                     ftype = LLAMA_FTYPE_ALL_F32;
                } break;
        }

        // this is a way to mark that we have "guessed" the file type
        ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

        {
            // adopted from upstream in this merge: get_key() (not a raw gguf_find_key) so a
            // kv_override for general.file_type is honored here too -- fork's original read
            // bypassed kv_overrides entirely for this one key.
            // ik_llama.cpp #2550 (MIT): a declared value that is not a known ftype keeps the content-derived guess instead of
            // overwriting it (the Bonsai-2 releases carry file_type values the loader does not name).
            uint32_t ftype_val = 0;
            if (get_key(LLM_KV_GENERAL_FILE_TYPE, ftype_val, false)) {
                if (llama_model_ftype_name((llama_ftype) ftype_val) != "unknown, may not work") {
                    ftype = (llama_ftype) ftype_val;
                }
            }
        }

        // The general.file_type read just above overwrites the content-derived guess
        // UNCONDITIONALLY, so a file that mis-declares itself is believed over its own tensors.
        // For the Bonsai types that is not hypothetical, MEASURED 2026-08-10 by reading both
        // GGUF headers directly:
        //     Ternary-Bonsai-27B-Q2_0.gguf  general.file_type=41  tensor types {42:498, F32:353}
        //     Bonsai-27B-Q1_0.gguf          general.file_type=40  tensor types {41:498, F32:353}
        // The vendor's numbering is off by one against ours, so llama-bench printed the 6.66 GiB
        // Q2_0 model as "Q1_0_G128 - 1.125 bpw" (it measures 2.126 bpw) and the Q1_0 model as
        // "unknown, may not work". Decoding was always correct -- the TENSOR types drive that --
        // but the label was not, and a mislabelled quant is how a reader reaches a wrong
        // conclusion about a file. Type-id false friends have cost this project real time twice.
        // Scoped to these two types on purpose so no other model's behaviour can change, and it
        // LOGS the disagreement rather than silently correcting it: a file lying about itself is
        // worth seeing, not merely worth fixing.
        if (type_max == GGML_TYPE_Q1_0_G128 || type_max == GGML_TYPE_Q2_0_G128) {
            const llama_ftype derived = (type_max == GGML_TYPE_Q1_0_G128)
                                      ? LLAMA_FTYPE_MOSTLY_Q1_0_G128
                                      : LLAMA_FTYPE_MOSTLY_Q2_0_G128;
            if (ftype != derived) {
                LLAMA_LOG_WARN("%s: general.file_type=%d disagrees with the dominant tensor type "
                               "(%s); trusting the tensors\n",
                               __func__, (int) ftype, ggml_type_name(type_max));
                ftype = derived;
            }
        }

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(meta, i);
            const enum gguf_type type   = gguf_get_kv_type(meta, i);
            // %zu (not %d) adopted from upstream: gguf_get_arr_n() returns size_t, and %d on a
            // 64-bit size_t is UB (and wrong on any array long enough to matter).
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(meta, i)), gguf_get_arr_n(meta, i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(meta, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    // adopted from upstream in this merge: gate the warning on use_mmap already being true --
    // fork's original warned even when the caller never asked for mmap in the first place.
    if (use_mmap && !llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        use_mmap = false;
    }
    if (repack_tensors) {
        use_mmap = false;
    }

    this->ncmoe = ncmoe;
    this->use_mmap = use_mmap;
    this->check_tensors = check_tensors;
    this->repack_tensors = repack_tensors;
    this->use_thp = use_thp;
    this->merge_qkv = merge_qkv;
    this->merge_up_gate_exps = merge_up_gate_exps;
    this->defer_experts = defer_experts;
    this->no_alloc = false; // this constructor always loads real tensor data
}

// UNIT 1 (S69-PORT-MODEL-INIT-FROM-USER-AND-FILE-PTR-LOADER-2026-09-28-01). Called ONLY by the two
// new ctors below -- the pre-existing fname ctor above keeps its own inline tail verbatim, byte for
// byte, so the byte-identical fname/path load this unit's control depends on is untouched by this
// refactor. This is the SAME logic as that inline tail (n_kv/n_tensors/fver, ftype detection over
// `weights`, the KV dump log, the trailing member assignments) with `fname` swapped for a
// caller-supplied source description string. An empty `weights` (the metadata-only ctor's case)
// makes the ftype-detection loop a no-op and ftype falls through to GGML_TYPE_F32 unless
// general.file_type is present in the metadata -- matches upstream's own externally-supplied-
// metadata branch (SOURCE-READ: upstream llama-model-loader.cpp ctor, `else` branch around line 710,
// which likewise does not populate its weights_map from caller metadata).
void llama_model_loader::finish_construction_from_meta(int ncmoe, bool use_mmap, bool check_tensors,
        bool repack_tensors, bool use_thp, bool merge_qkv, bool merge_up_gate_exps,
        bool defer_experts) {
    int trace = 0;
    if (getenv("LLAMA_TRACE")) {
        trace = atoi(getenv("LLAMA_TRACE"));
    }

    const char * source_desc = files.empty() ? "(metadata)" : "(file*)";

    if (import_nvfp4()) { // see the fname ctor: opt-in NVFP4 -> NVFP4_KS conversion
        use_mmap = false;
    }

    n_kv      = gguf_get_n_kv(meta);
    n_tensors = weights.size();

    fver = (enum llama_fver) gguf_get_version(meta);

    std::set<std::string> tensor_names;
    for (auto & w : weights) {
        n_elements += ggml_nelements(w.tensor);
        n_bytes    += ggml_nbytes(w.tensor);
        const std::string name(w.tensor->name);
        auto found = tensor_names.find(name);
        if (found != tensor_names.end()) {
            throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", w.tensor->name));
        }
        tensor_names.insert(name);
    }

    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
            __func__, n_kv, n_tensors, source_desc, llama_file_version_name(fver));

    // determine file type based on the number of tensors for each quantization and print meta data
    {
        std::map<enum ggml_type, uint32_t> n_type;

        uint32_t n_type_max = 0;
        enum ggml_type type_max = GGML_TYPE_F32;

        for (int i = 0; i < n_tensors; i++) {
            const ggml_tensor * tensor = weights.at(i).tensor;
            enum ggml_type type = tensor->type;

            // NVFP4 per-tensor scale companions ("<name>.scale", one F32 each) would otherwise outvote the real weights
            {
                const char * nm = ggml_get_name(tensor);
                const size_t ln = strlen(nm);
                if (type == GGML_TYPE_F32 && ln > 6 && strcmp(nm + ln - 6, ".scale") == 0 && ggml_nelements(tensor) == 1) {
                    continue;
                }
            }

            n_type[type]++;

            if (n_type_max < n_type[type]) {
                n_type_max = n_type[type];
                type_max   = type;
            }

            if (trace > 0) {
                const uint16_t sid = weights.at(i).idx;
                LLAMA_LOG_INFO("%s: - tensor %4d, split %2d: %32s %-8s [ %s ] %8.2f MiB\n", __func__, i, sid,
                        ggml_get_name(tensor), ggml_type_name(type), llama_format_tensor_shape(tensor).c_str(),
                        ggml_nbytes(tensor)/1024.0f/1024.0f);
            }
        }

        switch (type_max) {
            case GGML_TYPE_Q1_0_G128:ftype = LLAMA_FTYPE_MOSTLY_Q1_0_G128;break;
            case GGML_TYPE_Q2_0_G128:ftype = LLAMA_FTYPE_MOSTLY_Q2_0_G128;break;
            case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
            case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
            case GGML_TYPE_BF16:    ftype = LLAMA_FTYPE_MOSTLY_BF16;    break;
            case GGML_TYPE_TQ1_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ1_0;   break;
            case GGML_TYPE_TQ2_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ2_0;   break;
            case GGML_TYPE_BF16_R16:ftype = LLAMA_FTYPE_MOSTLY_BF16_R16;break;
            case GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
            case GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
            case GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
            case GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
            case GGML_TYPE_Q6_0:    ftype = LLAMA_FTYPE_MOSTLY_Q6_0;    break;
            case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
            case GGML_TYPE_Q8_KV:   ftype = LLAMA_FTYPE_MOSTLY_Q8_KV;   break;
            case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
            case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
            case GGML_TYPE_Q3_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q3_K_R4; break;
            case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
            case GGML_TYPE_Q4_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q4_K_R4; break;
            case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
            case GGML_TYPE_Q5_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q5_K_R4; break;
            case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
            case GGML_TYPE_Q6_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q6_K_R4; break;
            case GGML_TYPE_Q8_K_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_K_R8; break;
            case GGML_TYPE_Q8_KV_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_KV_R8; break;
            case GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
            case GGML_TYPE_IQ2_XXS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4; break;
            case GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
            case GGML_TYPE_IQ2_XS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS_R4; break;
            case GGML_TYPE_IQ2_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KS;  break;
            case GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_M;   break;
            case GGML_TYPE_IQ2_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_M_R4;break;
            case GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
            case GGML_TYPE_IQ3_XXS_R4: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4; break;
            case GGML_TYPE_IQ1_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ1_KT;  break;
            case GGML_TYPE_IQ2_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KT;  break;
            case GGML_TYPE_IQ3_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ3_KT;  break;
            case GGML_TYPE_IQ4_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_KT;  break;
            case GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
            case GGML_TYPE_IQ1_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ1_S_R4;break;
            case GGML_TYPE_IQ1_M_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ1_M_R4;break;
            case GGML_TYPE_IQ1_M:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;   break;
            case GGML_TYPE_IQ1_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ1_XS;  break;
            case GGML_TYPE_IQ1_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ1_XXS; break;
            case GGML_TYPE_IQ1_XXXS:ftype = LLAMA_FTYPE_MOSTLY_IQ1_XXXS;break;
            case GGML_TYPE_IQ1_BN:  ftype = LLAMA_FTYPE_MOSTLY_IQ1_BN;  break;
            case GGML_TYPE_IQ2_BN:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_BN;  break;
            case GGML_TYPE_IQ2_BN_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_BN_R4;break;
            case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
            case GGML_TYPE_IQ4_NL_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL_R4;break;
            case GGML_TYPE_IQ4_XS_R8:ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS_R8;break;
            case GGML_TYPE_Q4_0_R8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_R8; break;
            case GGML_TYPE_Q5_0_R4: ftype = LLAMA_FTYPE_MOSTLY_Q5_0_R4; break;
            case GGML_TYPE_Q6_0_R4: ftype = LLAMA_FTYPE_MOSTLY_Q6_0_R4; break;
            case GGML_TYPE_Q8_0_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_0_R8; break;
            case GGML_TYPE_MXFP4:   ftype = LLAMA_FTYPE_MOSTLY_MXFP4;   break;
            case GGML_TYPE_MXFP6_E2M3: ftype = LLAMA_FTYPE_MOSTLY_MXFP6_E2M3; break;
            case GGML_TYPE_MXFP6_E3M2: ftype = LLAMA_FTYPE_MOSTLY_MXFP6_E3M2; break;
            case GGML_TYPE_MXFP8_E4M3: ftype = LLAMA_FTYPE_MOSTLY_MXFP8_E4M3; break;
            case GGML_TYPE_MXFP8_E5M2: ftype = LLAMA_FTYPE_MOSTLY_MXFP8_E5M2; break;
            case GGML_TYPE_TQ3_1S:     ftype = LLAMA_FTYPE_MOSTLY_TQ3_1S;     break;
            case GGML_TYPE_TQ4_1S:     ftype = LLAMA_FTYPE_MOSTLY_TQ4_1S;     break;
            case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
            case GGML_TYPE_IQ4_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_KS;  break;
            case GGML_TYPE_IQ4_KS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_KS_R4;  break;
            case GGML_TYPE_IQ5_KS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ5_KS_R4;  break;
            case GGML_TYPE_IQ3_KS_R16:ftype = LLAMA_FTYPE_MOSTLY_IQ3_KS_R16;break;
            case GGML_TYPE_IQ4_KS_R16:ftype = LLAMA_FTYPE_MOSTLY_IQ4_KS_R16;break;
            case GGML_TYPE_IQ5_KS_R16:ftype = LLAMA_FTYPE_MOSTLY_IQ5_KS_R16;break;
            case GGML_TYPE_IQ4_KSS: ftype = LLAMA_FTYPE_MOSTLY_IQ4_KSS; break;
            case GGML_TYPE_IQ5_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ5_KS;  break;
            case GGML_TYPE_IQ2_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_K;   break;
            case GGML_TYPE_IQ2_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_K_R4;break;
            case GGML_TYPE_IQ3_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ3_KS;  break;
            case GGML_TYPE_IQ2_KL:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KL;  break;
            case GGML_TYPE_IQ3_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_K;   break;
            case GGML_TYPE_IQ3_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ3_K_R4;break;
            case GGML_TYPE_IQ4_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ4_K;   break;
            case GGML_TYPE_IQ4_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_K_R4;break;
            case GGML_TYPE_IQ5_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ5_K;   break;
            case GGML_TYPE_IQ5_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ5_K_R4;break;
            case GGML_TYPE_IQ6_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ6_K;   break;
            case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
            case GGML_TYPE_IQ3_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ3_S_R4;break;
            case GGML_TYPE_Q4_0_4_4: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_4_4; break;
            case GGML_TYPE_Q4_0_4_8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_4_8; break;
            case GGML_TYPE_Q4_0_8_8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_8_8; break;
            case GGML_TYPE_NVFP4:   ftype = LLAMA_FTYPE_MOSTLY_NVFP4;   break;
            case GGML_TYPE_NVFP4_KS:ftype = LLAMA_FTYPE_MOSTLY_NVFP4_KS;break;
            case GGML_TYPE_Q1_0:    ftype = LLAMA_FTYPE_MOSTLY_Q1_0;    break;
            case GGML_TYPE_Q2_0:    ftype = LLAMA_FTYPE_MOSTLY_Q2_0;    break;
            case GGML_TYPE_F8_E4M3: ftype = LLAMA_FTYPE_MOSTLY_F8_E4M3; break;
            default:
                {
                     LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                     ftype = LLAMA_FTYPE_ALL_F32;
                } break;
        }

        ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

        {
            uint32_t ftype_val = 0;
            if (get_key(LLM_KV_GENERAL_FILE_TYPE, ftype_val, false)) {
                ftype = (llama_ftype) ftype_val;
            }
        }

        if (type_max == GGML_TYPE_Q1_0_G128 || type_max == GGML_TYPE_Q2_0_G128) {
            const llama_ftype derived = (type_max == GGML_TYPE_Q1_0_G128)
                                      ? LLAMA_FTYPE_MOSTLY_Q1_0_G128
                                      : LLAMA_FTYPE_MOSTLY_Q2_0_G128;
            if (ftype != derived) {
                LLAMA_LOG_WARN("%s: general.file_type=%d disagrees with the dominant tensor type "
                               "(%s); trusting the tensors\n",
                               __func__, (int) ftype, ggml_type_name(type_max));
                ftype = derived;
            }
        }

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(meta, i);
            const enum gguf_type type   = gguf_get_kv_type(meta, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(meta, i)), gguf_get_arr_n(meta, i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(meta, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    if (use_mmap && !llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        use_mmap = false;
    }
    if (repack_tensors) {
        use_mmap = false;
    }

    this->ncmoe = ncmoe;
    this->use_mmap = use_mmap;
    this->check_tensors = check_tensors;
    this->repack_tensors = repack_tensors;
    this->use_thp = use_thp;
    this->merge_qkv = merge_qkv;
    this->merge_up_gate_exps = merge_up_gate_exps;
    this->defer_experts = defer_experts;
    this->no_alloc = false;
}

// UNIT 1: metadata-only construction. SOURCE-READ: upstream src/llama.cpp:446-457
// (llama_model_init_from_user pins params.load_mode=NONE, params.use_extra_bufts=false before
// calling down -- the equivalent pin for THIS fork's use_mmap is done here, unconditionally false,
// since there is no file to map).
llama_model_loader::llama_model_loader(struct gguf_context * metadata_in,
        llama_model_set_tensor_data_t set_tensor_data_in, void * set_tensor_data_ud_in,
        int ncmoe, bool check_tensors, bool repack_tensors, bool use_thp,
        bool merge_qkv, bool merge_up_gate_exps, bool defer_experts,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p) {
    GGML_ASSERT(metadata_in != nullptr);

    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            kv_overrides.insert({std::string(p->key), *p});
        }
    }
    tensor_buft_overrides = param_tensor_buft_overrides_p;

    set_tensor_data    = set_tensor_data_in;
    set_tensor_data_ud = set_tensor_data_ud_in;

    meta      = metadata_in;
    metadata  = meta;
    owns_meta = false; // caller-owned; ~llama_model_loader() must not free it

    get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
    llm_kv = LLM_KV(llm_arch_from_string(arch_name));
    llm_kv_use_file_prefix(llm_kv, arch_name);

    // No llama_file, no ggml_context, no weights/weights_map entries -- matches upstream's own
    // externally-supplied-metadata branch (see this method's header comment). `files` and
    // `contexts` stay empty; finish_construction_from_meta()'s ftype loop over an empty `weights`
    // is a documented no-op, not an oversight.
    // The merge / defer options probe the TYPES of tensors that exist in a file (require_tensor_meta in
    // llama-load-tensors.cpp); this source has none, so they are forced off here.
    finish_construction_from_meta(ncmoe, /*use_mmap*/ false, check_tensors, repack_tensors, use_thp,
            /*merge_qkv*/ false, /*merge_up_gate_exps*/ false, /*defer_experts*/ false);
}

// UNIT 1: FILE*-backed construction. SOURCE-READ: llama-mmap.cpp:415 (llama_file::llama_file(FILE*))
// and ggml/src/gguf.cpp:944 (gguf_init_from_file_ptr) -- both already existed in this fork before
// this unit; this ctor is the wiring the recon in the ledger predicted, reusing the SAME
// weights_map / create_tensor() path the fname ctor already uses. No split support (matches
// upstream's own scope for llama_model_load_from_file_ptr).
llama_model_loader::llama_model_loader(FILE * file, int ncmoe, bool use_mmap, bool check_tensors,
        bool repack_tensors, bool use_thp, bool merge_qkv, bool merge_up_gate_exps,
        bool defer_experts, const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p) {
    GGML_ASSERT(file != nullptr);

    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            kv_overrides.insert({std::string(p->key), *p});
        }
    }
    tensor_buft_overrides = param_tensor_buft_overrides_p;

    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    meta = gguf_init_from_file_ptr(file, params);
    if (!meta) {
        throw std::runtime_error(format("%s: failed to load model from file pointer\n", __func__));
    }
    metadata = meta;
    // owns_meta stays true (default): this ctor opened `meta` itself via gguf_init_from_file_ptr,
    // same ownership shape as the fname ctor's gguf_init_from_file -- the destructor must free it.

    get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
    llm_kv = LLM_KV(llm_arch_from_string(arch_name));
    llm_kv_use_file_prefix(llm_kv, arch_name);

    files.emplace_back(new llama_file(file));
    contexts.emplace_back(ctx);

    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        weights.emplace_back(files.back().get(), 0, cur->name, meta, cur);
        weights_map.emplace(cur->name, weights.back());
    }
    apply_ollama_compat();

    finish_construction_from_meta(ncmoe, use_mmap, check_tensors, repack_tensors, use_thp,
            merge_qkv, merge_up_gate_exps, defer_experts);
}

llama_model_loader::~llama_model_loader() {
    // owns_meta defaults to true, so `meta && owns_meta` == `meta` for every pre-existing ctor path
    // (fname): unchanged behaviour. UNIT 1: false only for the metadata-only ctor, whose `meta` is
    // caller-owned and must outlive this object.
    if (meta && owns_meta) {
        gguf_free(meta);
    }
    for (auto * ctx : contexts) {
        ggml_free(ctx);
    }
}

void llama_model_loader::build_expert_tensor_index(const llama_hparams & hparams) {
    expert_tensor_index = {};

    if (hparams.n_expert == 0 || hparams.n_layer == 0) {
        return;
    }

    expert_tensor_index.file_ranges.resize(files.size());

    size_t deferred_bytes = 0;
    const llm_arch arch = get_arch();

    for (const auto & weight : weights) {
        const std::string name(weight.tensor->name);
        uint32_t layer = 0;
        if (!parse_tensor_layer_index(name, layer)) {
            continue;
        }

        if (layer >= hparams.n_layer) {
            throw std::runtime_error(format("expert tensor '%s' has invalid layer index %u", name.c_str(), layer));
        }

        // check for split expert tensors (blk.N.ffn_gate.E.weight) by name pattern,
        // since llm_tensor_type can't resolve these (two %d in the format string)
        uint32_t expert = 0;
        if (is_split_expert_tensor(name, expert)) {
            if (expert >= hparams.n_expert) {
                throw std::runtime_error(format("expert tensor '%s' has invalid expert index %u", name.c_str(), expert));
            }
        } else {
            const llm_tensor tensor_type = llm_tensor_type(arch, name, int(layer));
            if (!is_merged_expert_tensor(tensor_type)) {
                continue;
            }
        }

        const size_t tensor_bytes = ggml_nbytes(weight.tensor);
        deferred_bytes += tensor_bytes;
        expert_tensor_index.file_ranges.at(weight.idx).push_back({ weight.offs, weight.offs + tensor_bytes });
    }

    for (auto & ranges : expert_tensor_index.file_ranges) {
        coalesce_ranges(ranges);
    }

    expert_tensor_index.deferred_bytes = deferred_bytes;
    expert_tensor_index.dense_bytes = n_bytes > deferred_bytes ? n_bytes - deferred_bytes : 0;
}

bool llama_model_loader::should_defer_expert_mmaps() const {
    return defer_experts && use_mmap && !expert_tensor_index.empty();
}

bool llama_model_loader::has_anonymous_mapping() const {
    for (const auto & mapping : mappings) {
        if (mapping->is_anonymous()) {
            return true;
        }
    }
    return false;
}

void llama_model_loader::drop_mmap_expert_pages() const {
    if (!use_mmap || mappings.empty() || expert_tensor_index.file_ranges.empty()) {
        return;
    }

    const size_t n_range_sets = std::min(mappings.size(), expert_tensor_index.file_ranges.size());
    for (size_t idx = 0; idx < n_range_sets; ++idx) {
        const auto & ranges = expert_tensor_index.file_ranges[idx];
        for (const auto & range : ranges) {
            mappings[idx]->dontneed_fragment(range.first, range.last);
        }
    }
}

void llama_model_loader::build_ple_tensor_index() {
    ple_tensor_index = {};
    ple_tensor_index.file_ranges.resize(files.size());
    auto index_range = [&](uint16_t idx, size_t offs, size_t nbytes) {
        ple_tensor_index.file_ranges.at(idx).push_back({ offs, offs + nbytes });
        ple_tensor_index.deferred_bytes += nbytes;
    };

    const auto * weight = get_weight(LLM_TN(get_arch())(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").c_str());
    if (weight != nullptr) {
        index_range(weight->idx, weight->offs, ggml_nbytes(weight->tensor));
    }

    // Engram tables are hash-indexed sparse lookups like PLE; the dense k/q/wkv stay prefetched.
    int n_engram = 0;
    for (int i = 0; i < n_tensors; ++i) {
        const std::string name = get_tensor_name(i);
        if (name.find("engram_embd.weight") == std::string::npos) {
            continue;
        }
        const auto * ew = get_weight(name.c_str());
        if (ew == nullptr) {
            continue;
        }
        index_range(ew->idx, ew->offs, ggml_nbytes(ew->tensor));
        ++n_engram;
    }
    if (n_engram > 0) {
        LLAMA_LOG_INFO("%s: indexed %d engram tables for deferred loading\n", __func__, n_engram);
    }
}

void llama_model_loader::index_lazy_flagged_tensors() {
    if (lazy_flagged.empty() || (!defer_ple && !defer_ple_auto)) {
        return; // --lazy-mode off, or nothing flagged
    }
    constexpr size_t auto_min_size = 4ull * 1024 * 1024 * 1024; // upstream lazy_read::add auto_min_size
    if (ple_tensor_index.file_ranges.size() != files.size()) {
        ple_tensor_index.file_ranges.resize(files.size());
    }
    for (const auto & name : lazy_flagged) {
        const auto * w = get_weight(name.c_str());
        if (w == nullptr || w->idx >= ple_tensor_index.file_ranges.size()) {
            continue;
        }
        const size_t nbytes = ggml_nbytes(w->tensor);
        if (ple_deferred_bytes_in(w->idx, w->offs, w->offs + nbytes) == nbytes) {
            continue; // already indexed by name
        }
        if (!defer_ple && nbytes <= auto_min_size) {
            continue; // auto: small tensors stay resident
        }
        ple_tensor_index.file_ranges.at(w->idx).push_back({ w->offs, w->offs + nbytes });
        ple_tensor_index.deferred_bytes += nbytes;
        defer_ple = true; // auto crossed the threshold
        LLAMA_LOG_INFO("%s: tensor %s (size = %zu MiB) lazy read enabled by TENSOR_READ_LAZY\n",
                __func__, name.c_str(), nbytes / 1024 / 1024);
    }
}

bool llama_model_loader::should_defer_ple_mmaps() const {
    return defer_ple && use_mmap && !ple_tensor_index.empty();
}

void llama_model_loader::apply_ple_mmap_policy() const {
    for (size_t idx = 0; idx < ple_tensor_index.file_ranges.size(); ++idx) {
        for (const auto & range : ple_tensor_index.file_ranges[idx]) {
            // sparse row lookups through get_rows: readahead would evict more than it fetches
            mappings[idx]->random_fragment(range.first, range.last);
            mappings[idx]->dontneed_fragment(range.first, range.last);
        }
    }
}

bool llama_model_loader::file_has_deferred_ple(int idx) const {
    if (idx < 0 || (size_t) idx >= ple_tensor_index.file_ranges.size()) {
        return false;
    }
    for (const auto & range : ple_tensor_index.file_ranges[(size_t) idx]) {
        if (!range.empty()) {
            return true;
        }
    }
    return false;
}

bool llama_model_loader::ple_range_overlaps(int idx, size_t first, size_t last) const {
    return ple_deferred_bytes_in(idx, first, last) > 0;
}

size_t llama_model_loader::ple_deferred_bytes_in(int idx, size_t first, size_t last) const {
    if (idx < 0 || (size_t) idx >= ple_tensor_index.file_ranges.size() || first >= last) {
        return 0;
    }
    size_t bytes = 0;
    for (const auto & range : ple_tensor_index.file_ranges[(size_t) idx]) {
        const size_t lo = std::max(first, range.first);
        const size_t hi = std::min(last, range.last);
        if (hi > lo) {
            bytes += hi - lo;
        }
    }
    return bytes;
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
llama_model_loader::get_arr_n(const std::string & key, T & result, const bool required) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }
        return false;
    }

    struct GGUFMeta::ArrayInfo arr_info =
        GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);


    result = arr_info.length;
    return true;
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
llama_model_loader::get_arr_n(const enum llm_kv kid, T & result, const bool required) {
    return get_arr_n(llm_kv(kid), result, required);
}

template<typename T>
bool llama_model_loader::get_arr(const std::string & key, std::vector<T> & result, const bool required) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY) {
        if (required) {
            throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
        }
        return false;
    }

    if constexpr (std::is_same_v<T, std::string>) {
        if (gguf_get_arr_type(meta, kid) != GGUF_TYPE_STRING) {
            throw std::runtime_error(format("%s is not a string array", key.c_str()));
        }
        const size_t n = gguf_get_arr_n(meta, kid);
        result.clear();
        result.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            result.emplace_back(gguf_get_arr_str(meta, kid, i));
        }
        return true;
    } else {
    struct GGUFMeta::ArrayInfo arr_info =
        GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);

    switch (arr_info.gt) {
        case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same<T, float>::value)); break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_BOOL:
        case GGUF_TYPE_INT32:   GGML_ASSERT((std::is_same_v<T,  int32_t>) || (std::is_same_v<T, uint32_t>));  break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:   GGML_ASSERT((std::is_same_v<T,  int64_t>) || (std::is_same_v<T, uint64_t>));  break;
        default:
            throw std::runtime_error(format("%s is not a float32, int32, uint32, int64, uint64 or bool array", key.c_str()));
    }

    result.resize(arr_info.length);
    if (arr_info.gt == GGUF_TYPE_BOOL) {
        std::transform((const int8_t *)arr_info.data, (const int8_t *)arr_info.data + arr_info.length, result.begin(),
                [] (int8_t x) { return static_cast<T>(x != 0); });

    } else {
        result.assign((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length);
    }

    return true;
    } // numeric arrays
}

template<typename T, size_t N_MAX>
bool llama_model_loader::get_arr(const std::string & key, std::array<T, N_MAX> & result, const bool required) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY) {
        if (required) {
            throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
        }
        return false;
    }

    struct GGUFMeta::ArrayInfo arr_info =
        GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);

    switch (arr_info.gt) {
        case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same_v<T, float>)); break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:   GGML_ASSERT((std::is_same_v<T,  int32_t>) || (std::is_same_v<T, uint32_t>) || (std::is_same_v<T, bool>));  break;
        // a bool array may be read into integer OR bool elements (minimax-01 reads attention.recurrent_layers into std::array<bool, N>)
        case GGUF_TYPE_BOOL:    GGML_ASSERT((std::is_same_v<T,  int32_t>) || (std::is_same_v<T, uint32_t>) || (std::is_same_v<T, bool>));  break;
        default:
            throw std::runtime_error(format("%s is not a float32, int32 array", key.c_str()));
    }

    if (arr_info.length > N_MAX) {
        throw std::runtime_error(format("array length %u for key %s exceeds max %u", (uint32_t) arr_info.length, key.c_str(), (uint32_t) N_MAX));
    }

    if (arr_info.gt == GGUF_TYPE_BOOL) {
        std::transform((const int8_t *)arr_info.data, (const int8_t *)arr_info.data + arr_info.length, result.begin(),
                [] (int8_t x) { return static_cast<T>(x != 0); });
    } else if constexpr (std::is_same_v<T, bool>) {
        // an integer array read into bool elements (spark2_5 sliding_window_pattern): non-zero is true
        std::transform((const int32_t *)arr_info.data, (const int32_t *)arr_info.data + arr_info.length, result.begin(),
                [] (int32_t x) { return x != 0; });
    } else {
        std::copy((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length, result.begin());
    }

    return true;
}

template<typename T>
bool llama_model_loader::get_arr(const enum llm_kv kid, T & result, const bool required) {
    return get_arr(llm_kv(kid), result, required);
}

template<typename T>
bool llama_model_loader::get_key(const std::string & key, T & result, const bool required) {
    auto it = kv_overrides.find(key);

    const struct llama_model_kv_override * override =
        it != kv_overrides.end() ? &it->second : nullptr;

    const bool found = GGUFMeta::GKV<T>::set(meta, key, result, override);

    if (required && !found) {
        throw std::runtime_error(format("key not found in model: %s", key.c_str()));
    }

    return found;
}

template<typename T>
bool llama_model_loader::get_key(const enum llm_kv kid, T & result, const bool required) {
    return get_key(llm_kv(kid), result, required);
}

// get array of n <= N_MAX elements, or a single element repeated n times
template<typename T, size_t N_MAX>
bool llama_model_loader::get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, const bool required,
                                        const bool allow_partial) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }
        return false;
    }

    if (n > N_MAX) {
        throw std::runtime_error(format("n > N_MAX: %u > %u for key %s", (uint32_t) n, (uint32_t) N_MAX, key.c_str()));
    }

    if (gguf_get_kv_type(meta, kid) == GGUF_TYPE_ARRAY) {
        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);

        if (n != arr_info.length) {
            // tolerate a SHORTER stored array (zero-pad the rest) when the caller opts in -- see the header.
            if (allow_partial && arr_info.length < n) {
                result.fill(T(0));                 // get_arr copies arr_info.length elements; the rest stay 0
                return get_arr(key, result, required);
            }
            throw std::runtime_error(format("key %s has wrong array length; expected %u, got %u", key.c_str(), n, (uint32_t) arr_info.length));
        }

        return get_arr(key, result, required);
    } else {
        T value;

        bool ok = get_key(key, value, required);
        if (!ok) {
            return false;
        }

        for (uint32_t i = 0; i < n; i++) {
            result[i] = value;
        }

        return true;
    }
}

template<typename T>
bool llama_model_loader::get_key_or_arr(const enum llm_kv kid, T & result, uint32_t n, const bool required,
                                        const bool allow_partial) {
    return get_key_or_arr(llm_kv(kid), result, n, required, allow_partial);
}

void llama_model_loader::get_mrope_sections(std::array<int, 4> & sections, const std::array<int, 4> * fallback) {
    sections.fill(0);
    bool found = get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, sections, 4, false, true);
    if (!found) {
        // Ollama's spelling of the same array: "<arch>.rope.mrope_section" (read by its engine at
        // model/models/qwen25vl/model_text.go:47; qwen35 files carry both spellings).
        std::string alt = llm_kv(LLM_KV_ROPE_DIMENSION_SECTIONS);
        const std::string tail = "rope.dimension_sections";
        if (alt.size() > tail.size() && alt.compare(alt.size() - tail.size(), tail.size(), tail) == 0) {
            alt.replace(alt.size() - tail.size(), tail.size(), "rope.mrope_section");
            found = get_key_or_arr(alt, sections, 4, false, true);   // allow_partial: Ollama writes 3 entries
            if (found) {
                LLAMA_LOG_INFO("%s: M-RoPE sections read from %s (Ollama spelling)\n", __func__, alt.c_str());
            }
        }
    }
    const int  sum   = sections[0] + sections[1] + sections[2] + sections[3];
    if (sum > 0) {
        return;
    }
    const std::string key = llm_kv(LLM_KV_ROPE_DIMENSION_SECTIONS);
    if (fallback == nullptr) {
        throw std::runtime_error(format("%s %s: this arch always applies M-RoPE and has no sourced default; "
                                        "re-convert with convert_hf_to_gguf.py (it writes the key from config.json)",
                                        found ? "all-zero" : "missing", key.c_str()));
    }
    sections = *fallback;
    LLAMA_LOG_WARN("%s: %s %s -- using the %s default [%d, %d, %d, %d] (HF config.json mrope_section)\n", __func__,
                   found ? "all-zero" : "missing", key.c_str(), arch_name.c_str(),
                   sections[0], sections[1], sections[2], sections[3]);
}

bool llama_model_loader::get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required) {
    const std::string key = llm_kv(kid);

    const int id = gguf_find_key(meta, key.c_str());

    if (id < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }
        return false;
    }

    // throw and error if type is an array
    if (gguf_get_kv_type(meta, id) == GGUF_TYPE_ARRAY) {
        if (required) {
            throw std::runtime_error(format("expected scalar, found array for key: %s", key.c_str()));
        }
        return false;
    }

    return get_key(key, result, required);
}


const char * llama_model_loader::get_tensor_name(int i) const {
    return weights.at(i).tensor->name;
}

// adopted from upstream in this merge: O(log n) weights_map lookup, replacing fork's original
// O(n) linear scan over `weights` -- safe now that weights_map is populated in lock-step with
// `weights` in the constructor (see there).
const llama_model_loader::llama_tensor_weight * llama_model_loader::get_weight(const char * name) const {
    auto pos = weights_map.find(name);
    if (pos != weights_map.end()) {
        return &pos->second;
    }
    // Tensor-name aliases for GGUFs written by Ollama's converter (MIT), consulted ONLY when the name we
    // ask for is absent, so a file in llama.cpp naming is unaffected. Each entry is the same tensor under
    // another name, same layout -- verified against both converters, not assumed:
    //   <blk>.ssm_dt.bias -> <blk>.ssm_dt   qwen35 dt_bias. Ollama convert_qwen3next.go:928 maps
    //       linear_attn.dt_bias to "ssm_dt" with no suffix; upstream conversion/qwen.py writes ssm_dt.bias.
    //       Both apply the same grouped->tiled V-head reorder (Ollama :467-471, upstream qwen.py:596-603).
    //       A mamba-family file always carries .bias, so this never shadows a real tensor.
    static const std::pair<const char *, const char *> k_suffix_alias[] = {
        { ".ssm_dt.bias", ".ssm_dt" },
    };
    const size_t len = strlen(name);
    for (const auto & [ours, theirs] : k_suffix_alias) {
        const size_t lo = strlen(ours);
        if (len > lo && strcmp(name + len - lo, ours) == 0) {
            const std::string alt = std::string(name, len - lo) + theirs;
            pos = weights_map.find(alt);
            if (pos != weights_map.end()) {
                return &pos->second;
            }
        }
    }
    return nullptr;
}

const llama_model_loader::llama_tensor_weight & llama_model_loader::require_weight(const char * name) const {
    const llama_tensor_weight * weight = get_weight(name);
    if (!weight) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
    }
    return *weight;
}

struct ggml_tensor * llama_model_loader::get_tensor_meta(const char * name) const {
    const auto * weight = get_weight(name);
    if (!weight) {
        return nullptr;
    }
    return weight->tensor;
}

// header declares an additional inline `const std::string&` overload that forwards here --
// upstream's separate std::string-taking definition of this same function is now redundant.
struct ggml_tensor * llama_model_loader::require_tensor_meta(const char * name) const {
    struct ggml_tensor * tensor = get_tensor_meta(name);
    if (!tensor) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
    }
    return tensor;
}

struct ggml_tensor * llama_model_loader::create_tensor_for(struct ggml_context * ctx, const struct ggml_tensor * cur, bool duplicated) {
    struct ggml_tensor * tensor = ggml_dup_tensor(ctx, cur);
    ggml_set_name(tensor, ggml_get_name(cur));

    if (duplicated) {
        size_data += ggml_nbytes(cur);
    } else {
        n_created++;
        if (!created_names.insert(ggml_get_name(cur)).second) {
            created_twice.emplace_back(ggml_get_name(cur));
        }
    }

    return tensor;
}

// `allow_reshape` adopted from upstream in this merge (TENSOR_ALLOW_RESHAPE); default false in
// the header preserves every existing 3-arg call site's exact behavior.
const struct ggml_tensor * llama_model_loader::check_tensor_dims(
        const std::string & name, const std::vector<int64_t> & ne, bool required, bool allow_reshape) const {
    const struct ggml_tensor * cur = get_tensor_meta(name.c_str());

    if (cur == NULL) {
        if (!required) {
            return NULL;
        }
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }

    {
        bool is_ok = true;

        if (allow_reshape) {
            // check total number of elements only
            const int64_t ncur = ggml_nelements(cur);
            int64_t nexp = 1;
            for (size_t i = 0; i < ne.size(); ++i) {
                nexp *= ne[i];
            }
            if (ncur != nexp) {
                is_ok = false;
            }
        } else {
            for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
                if ((i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)) {
                    is_ok = false;
                    break;
                }
            }
        }

        if (!is_ok) {
            throw std::runtime_error(
                    format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                        __func__, name.c_str(),
                        llama_format_tensor_shape(ne).c_str(),
                        llama_format_tensor_shape(cur).c_str()));
        }
    }

    return cur;
}

// A draft head that shares the target's embeddings (a NextN/MTP sidecar converted with
// --mtp-shared-embd) ships neither token_embd, output_norm nor the lm head; resolve those three names
// against the already loaded target instead.
// SOURCE-READ: unslothai/llama.cpp f36eda84 src/llama-model-loader.cpp borrow_shared_tensor (MIT).
// Mapped onto THIS loader: its create_tensor() takes a name string rather than an LLM_TN_IMPL, so the
// three roles are matched by their fixed GGUF names, and the target's tensor is read from the
// llama_model members (tok_embd / output_norm / output) instead of searching tensors_by_name: a target
// whose head is tied keeps `output` as a duplicate of token_embd there, which is the tensor its own
// graph reads, and a name search would find the input copy instead.
// Returns nullptr when nothing is borrowed, so every other tensor of every model is unaffected.
struct ggml_tensor * llama_model_loader::borrow_shared_tensor(
        const std::string & name, const std::vector<int64_t> & ne, int flags) {
    if (name != "token_embd.weight" && name != "output_norm.weight" && name != "output.weight") {
        return nullptr;
    }
    if (flags & TENSOR_SKIP) {
        return nullptr;
    }
    // a metadata-only source has no weights_map by design, so the 'absent from the file' tests below would misread
    // every tensor as missing; it also has no target model to borrow from
    if (set_tensor_data) {
        return nullptr;
    }
    // a file that ships the tensor keeps its own copy
    if (get_weight(name.c_str()) != nullptr) {
        return nullptr;
    }
    // a draft that left out its embeddings left out the whole set, so the missing token_embd is what
    // marks one. Without this an arch that ties the head to its own token_embd when output.weight is
    // absent would borrow the target's head instead of using its own embedding.
    if (get_weight("token_embd.weight") != nullptr) {
        return nullptr;
    }

    if (model_shared == nullptr) {
        throw std::runtime_error(format("%s: missing tensor '%s'; if this is a draft head that shares the target's "
                    "embeddings, load it as a draft of its target model, not on its own", __func__, name.c_str()));
    }

    ggml_tensor * src = nullptr;
    if (name == "token_embd.weight")      src = model_shared->tok_embd;
    else if (name == "output_norm.weight") src = model_shared->output_norm;
    else                                   src = model_shared->output;

    if (src == nullptr) {
        if (flags & TENSOR_NOT_REQUIRED) {
            return nullptr;   // the target has none either (e.g. no separate head): the caller falls back as it would alone
        }
        throw std::runtime_error(format("%s: draft needs tensor '%s' from the target, which does not have it", __func__, name.c_str()));
    }
    if (src->extra != nullptr) {
        throw std::runtime_error(format("%s: target tensor '%s' is split across devices; a draft cannot share it", __func__, name.c_str()));
    }

    // the draft uses the tensor directly, so the shapes must agree exactly
    bool same = true;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const int64_t want = d < (int) ne.size() ? ne[d] : 1;
        if (src->ne[d] != want) {
            same = false;
        }
    }
    if (!same) {
        throw std::runtime_error(format("%s: draft and target disagree on '%s': target has %s, draft wants %s",
                    __func__, name.c_str(), llama_format_tensor_shape(src).c_str(), llama_format_tensor_shape(ne).c_str()));
    }

    LLAMA_LOG_INFO("%s: tensor %s taken from the target model\n", __func__, name.c_str());

    // not counted in n_created or size_data: the tensor is not in this file and is neither allocated
    // nor freed here
    return src;
}

struct ggml_tensor * llama_model_loader::create_tensor(struct ggml_context * ctx, const std::string & name,
        const std::vector<int64_t> & ne, int flags) {
    // Metadata-only source (llama_model_init_from_user): the constructor created no weights, so a REQUIRED tensor the
    // caller's metadata does not describe is created here as F32 with the shape the architecture asks for, and
    // the caller's set_tensor_data callback fills it. SOURCE-READ: the F32 default upstream's buft_list-aware
    // create_tensor() applies on this path (see the ctor note in llama-model-loader.h). Optional tensors stay absent.
    // Optional tensors (TENSOR_NOT_REQUIRED) ARE created here as well, despite what the sentence above says: restricting the fallback to
    // required tensors was tried 2026-10-04 and made llama, qwen35, phi3, ernie4_5-moe and mistral3 stop building / crash (0xC0000005),
    // because those graph builders dereference an optional tensor unconditionally and a real model file carries it. Names the architecture
    // does not map come out as "__missing__" (llama-model.cpp:2707-2737): those are absent on a real file too, so they are NOT created here
    // (several of them with different shapes cannot live under one name - MEASURED 2026-10-04: a saved model failed to reload with
    // "check_tensor_dims: tensor '__missing__' has wrong shape" for granite).
    // TENSOR_SKIP_IF_VIRTUAL: an optional alternative layout (the fused QKV in create_tensor_qkv) that a metadata-only model must NOT
    // conjure, or the class loader takes the fused branch and leaves wq/wk/wv null (olmo2, olmoe: ggml_mul_mat(a = NULL) at graph build).
    // Upstream's metadata-only path skips these the same way; the legacy creators never pass this flag.
    if (set_tensor_data && (flags & TENSOR_SKIP_IF_VIRTUAL) && get_weight(name.c_str()) == nullptr) {
        return nullptr;
    }
    // An optional tensor with a zero-sized dimension (an expert count of 0 on a dense layer of a hybrid arch: jamba's ffn_gate_inp) is "absent",
    // not an empty tensor: the class loader decides dense vs MoE by whether the pointer is set, and an empty one sent jamba into build_moe_ffn.
    if (set_tensor_data && (flags & TENSOR_NOT_REQUIRED) && get_weight(name.c_str()) == nullptr) {
        for (int64_t d : ne) {
            if (d == 0) {
                return nullptr;
            }
        }
    }
    if (set_tensor_data && !(flags & TENSOR_SKIP) && name != "__missing__" && get_weight(name.c_str()) == nullptr) {
        struct ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) ne.size(), ne.data());
        ggml_set_name(t, name.c_str());
        size_data += ggml_nbytes(t);
        if (!(flags & TENSOR_DUPLICATED)) {
            n_created++;
        }
        return t;
    }

    // must run before check_tensor_dims: the tensor is absent from this file by design
    if (ggml_tensor * shared = borrow_shared_tensor(name, ne, flags)) {
        return shared;
    }

    const struct ggml_tensor * cur = check_tensor_dims(name, ne, !(flags & TENSOR_NOT_REQUIRED));

    if (cur == NULL) {
        return NULL;
    }

    // skip unused tensors (per-tensor detail at debug level; one summary at
    // done_getting_tensors so ordinary loads are not flooded by an unused
    // NextN/MTP head)
    if (flags & TENSOR_SKIP) {
        const size_t nbytes = ggml_nbytes(cur);
        LLAMA_LOG_DEBUG("model has unused tensor %s (size = %zu bytes) -- ignoring\n", name.c_str(), nbytes);

        n_skipped++;
        size_skipped += nbytes;

        size_data -= nbytes;
        n_created++;
        if (!created_names.insert(name).second) {
            created_twice.emplace_back(name + " (skipped)");
        }

        return nullptr;
    }

    if ((flags & TENSOR_READ_LAZY) && lazy_flagged.size() < 4096) {
        lazy_flagged.push_back(name);
    }

    return create_tensor_for(ctx, cur, flags & TENSOR_DUPLICATED);
}

struct ggml_tensor * llama_model_loader::create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base,
        const std::string & name, const std::vector<int64_t> & ne, size_t offset, bool required) {
    const struct ggml_tensor * cur = check_tensor_dims(name, ne, required);

    if (cur == NULL) {
        return NULL;
    }

    if (cur->type != base->type) {
        throw std::runtime_error(format("%s: tensor '%s' has wrong type; expected %s, got %s", __func__, name.c_str(), ggml_type_name(base->type), ggml_type_name(cur->type)));
    }

    std::array<int64_t, GGML_MAX_DIMS> dims;
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        dims[i] = i < ne.size() ? ne[i] : 1;
    }

    struct ggml_tensor * tensor = ggml_view_4d(ctx, base,
            dims[0], dims[1], dims[2], dims[3],
            cur->nb[1], cur->nb[2], cur->nb[3],
            offset);

    ggml_set_name(tensor, name.c_str());

    n_created++;

    return tensor;
}

// adopted from upstream in this merge: `partial` (default false) allows n_created < n_tensors
// when a caller explicitly opts in (e.g. two models sharing one .gguf); default preserves the
// original strict exact-match behavior. n_tensors_moved diagnostic NOT carried -- it belongs to
// the buft_list-driven create_tensor overload excluded above (see that block's note); nothing
// in this file's live create_tensor() ever increments it.
void llama_model_loader::done_getting_tensors(bool partial) const {
    if (n_skipped > 0) {
        LLAMA_LOG_WARN("%s: skipped %d unused tensors (%.1f MiB) -- e.g. a NextN/MTP head when MTP is not requested; per-tensor detail at debug log level\n",
                __func__, n_skipped, size_skipped / 1024.0 / 1024.0);
    }
    if (set_tensor_data) {
        return;   // metadata-only source: n_tensors is 0 by construction, so the count checks below do not apply
    }
    if (n_created > n_tensors) {
        std::string twice;
        for (size_t i = 0; i < created_twice.size() && i < 16; ++i) {
            twice += (i ? ", " : "") + created_twice[i];
        }
        throw std::runtime_error(format("%s: too many tensors created; expected %d, got %d; created more than once (%zu): %s",
                __func__, n_tensors, n_created, created_twice.size(), twice.empty() ? "(none: the excess is not a repeat)" : twice.c_str()));
    }
    if (n_created < n_tensors && !partial) {
        // Combined vision GGUF (e.g. qwen3vl from Ollama) carries the vision tower (v.* / mm.*) in the SAME file
        // as the text model. Those tensors belong to the separate clip/mmproj loader (--mmproj), so they stay
        // unclaimed here. Do not fail when the ONLY unclaimed tensors are vision tensors -- the text model then
        // loads from a combined VL GGUF. Was: "wrong number of tensors; expected 858, got 399".
        // DEC-S69-VL-COMBINED-GGUF-VISION-SKIP; recovered 2026-09-27 from stranded commit 3e367ef6b.
        int n_vision = 0;
        int n_mtp    = 0;   // Ollama writes a qwen35 MTP head as mtp.* (ours: blk.N.nextn.*); not mapped yet
        for (int i = 0; i < n_tensors; ++i) {
            const char * nm = get_tensor_name(i);
            if (nm && (strncmp(nm, "v.", 2) == 0 || strncmp(nm, "mm.", 3) == 0)) {
                n_vision++;
            } else if (nm && strncmp(nm, "mtp.", 4) == 0) {
                n_mtp++;
            }
        }
        if ((n_vision > 0 || n_mtp > 0) && n_created == n_tensors - n_vision - n_mtp) {
            LLAMA_LOG_INFO("%s: combined GGUF - %d text tensors loaded, %d vision tensors left to mmproj/clip, "
                           "%d Ollama-named MTP tensors (mtp.*) left unused\n", __func__, n_created, n_vision, n_mtp);
            return;
        }
        throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d (vision tensors in file: %d, mtp.*: %d)",
                                        __func__, n_tensors, n_created, n_vision, n_mtp));
    }
    if (n_created < n_tensors) {
        LLAMA_LOG_INFO("%s: partial load - used %d of %d tensors in the file (rest belong to a sibling model on the same .gguf)\n",
                __func__, n_created, n_tensors);
    }
}

void llama_model_loader::init_mappings(bool prefetch, llama_mlocks * mlock_mmaps, bool use_thp) {
    if (use_mmap) {
        mappings.reserve(files.size());
        mmaps_used.reserve(files.size());
        size_t n_deferred = 0;
        for (size_t fi = 0; fi < files.size(); ++fi) {
            const auto & file = files[fi];
            // Deferred sparse tables (PLE, engram) stay cold; faulting GBs upfront wastes IO.
            const bool deferred_file = file_has_deferred_ple((int) fi);
            if (deferred_file) {
                ++n_deferred;
            }
            const size_t file_prefetch = (prefetch && !deferred_file) ? (size_t) -1 : 0;
            std::unique_ptr<llama_mmap> mapping(new llama_mmap(file.get(), file_prefetch, ggml_is_numa(), use_thp));
            mmaps_used.emplace_back(mapping->size(), 0);
            if (mlock_mmaps) {
                std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                mlock_mmap->init(mapping->addr());
                mlock_mmaps->emplace_back(std::move(mlock_mmap));
            }
            mappings.emplace_back(std::move(mapping));
        }
        if (n_deferred > 0) {
            LLAMA_LOG_INFO("%s: %zu of %zu files hold deferred tables and skip bulk prefetch\n",
                    __func__, n_deferred, files.size());
        }
    }

    // compute the total size of all tensors for progress reporting
    for (auto & w : weights) {
        size_data += ggml_nbytes(w.tensor);
    }
}

using mapping_ranges = std::vector<llama_model_loader::mapping_range>;

// (merge 2026-11-04: an auto-merge hybrid of upstream get_mapping_ranges(idx, ctx) and the fork get_mapping_range
//  body was removed here, with the static merge_ranges / sub_ranges helpers only it used; see batch 39.)

// the fork's single-span form, restored 2026-11-05 from HEAD (src/llama-model-loader.cpp:2149-2171 there)
void llama_model_loader::get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const {
    GGML_ASSERT(!mappings.empty());
    const auto & mapping = mappings.at(idx);

    *first = mapping->size();
    *last  = 0;
    *addr = mapping->addr();
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        try {
            const auto * weight = get_weight(ggml_get_name(tensor));
            if (!weight) {
                continue;
            }
            if (weight->idx != idx) {
                continue;
            }
            *first = std::min(*first, weight->offs);
            *last  = std::max(*last,  weight->offs + ggml_nbytes(tensor));
        } catch(...) {
            // the tensor is not in the model
        }
    }
}


// adopted from upstream in this merge: real, working method. Confirmed live:
// src/llama-quant.cpp does `if (ml.use_mmap) { ml.unmap_weight(weight); }`. Reuses the SAME
// llama_mmap primitive drop_mmap_expert_pages() below uses (dontneed_fragment), but calls the
// stronger unmap_fragment -- upstream's choice, and the right one here: a quantize pass is done
// with the weight for good (not "might defer-load it later"), so a full unmap is correct where
// drop_mmap_expert_pages()'s softer "don't need this yet" hint would not be.
void llama_model_loader::unmap_weight(const llama_tensor_weight & w) const {
    if (!use_mmap) {
        return;
    }
    mappings.at(w.idx)->unmap_fragment(w.offs, w.offs + ggml_nbytes(w.tensor));
}

// ADOPTED from unslothai/llama.cpp PR #152 ("llama: map each contiguous run of a context's
// tensors, not one span over all of them"), one of the 14 items pinned into their
// b10715-mix-86bd2d3 nightly. Ported to this fork's pre-refactor loader; upstream applies it
// in llama-model.cpp::llama_model_base::load_tensors, ours lives in llama.cpp, but
// get_mapping_range / llama_buf_map / load_all_data are structurally identical here.
// COMPATIBILITY-INVESTIGATED: three shapes were compared before porting -- upstream's flat
//   span, this fork's get_mapping_range, and the PR's merged-run form. The fork keeps its own
//   exception handling (below) rather than taking upstream's loop verbatim, so the two
//   divergent designs are combined rather than one replacing the other.
// WEB-RESEARCH-DONE: https://github.com/unslothai/llama.cpp/pull/152
//
// get_mapping_range returns ONE span [min offs, max offs). If a tensor belonging to a
// DIFFERENT buffer type sits between this context's tensors, that single span covers it too,
// so the foreign tensor is mapped into this context's backend buffer as well. This walks the
// tensors, sorts them, and merges only runs that are genuinely near-adjacent.
//
// The try/catch around get_weight is carried over from this fork's get_mapping_range, where
// get_weight can throw for a tensor that is not in the model; upstream's variant has no such
// call shape, so the guard is additive here.
void llama_model_loader::get_mapping_ranges(std::vector<std::pair<size_t, size_t>> & ranges, void ** addr, int idx, ggml_context * ctx) const {
    GGML_ASSERT(!mappings.empty());
    const auto & mapping = mappings.at(idx);

    ranges.clear();
    *addr = mapping->addr();

    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        try {
            const auto * weight = get_weight(ggml_get_name(tensor));
            if (!weight || weight->idx != idx) {
                continue;
            }
            ranges.emplace_back(weight->offs, weight->offs + ggml_nbytes(tensor));
        } catch(...) {
            // the tensor is not in the model
        }
    }

    if (ranges.empty()) {
        return;
    }

    std::sort(ranges.begin(), ranges.end());

    // below this a gap is padding, and splitting on it would cost a buffer for nothing
    constexpr size_t min_gap = 32ull*1024*1024;

    // never trade one oversized buffer for very many small ones
    constexpr size_t max_ranges = 64;

    size_t n = 0;
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first > ranges[n].second + min_gap) {
            ranges[++n] = ranges[i];
        } else {
            ranges[n].second = std::max(ranges[n].second, ranges[i].second);
        }
    }
    ranges.resize(n + 1);

    if (ranges.size() > max_ranges) {
        ranges = { { ranges.front().first, ranges.back().second } };
    }
}

// Mainline NVFP4 GGUF (type 40 + per-tensor `.scale`) -> NVFP4_KS (type 168, the scale folded into a per-row fp32).
// SOURCE-READ: ik_llama.cpp 30903876b src/llama-model-loader.cpp import_nvfp4/read_nvfp4_rows/load_nvfp4_tensor (MIT).
// DIFFERENCE FROM ik: ik always converts (it has no native type 40). This fork has one (CPU iqk kernel, CUDA mmvq +
// dequant/cuBLAS, W4A4 prefill, graph-side `.scale` MUL), so the conversion is OFF unless LLAMA_NVFP4_IMPORT_KS is
// set to a non-"0" value, and a default load of a type-40 GGUF is byte-for-byte what it was.
static bool nvfp4_import_enabled() {
    const char * e = getenv("LLAMA_NVFP4_IMPORT_KS");
    return e && *e && strcmp(e, "0") != 0;
}

bool llama_model_loader::import_nvfp4() {
    // SOURCE-READ: ik_llama.cpp c7ea52b33 src/llama-model-loader.cpp import_nvfp4/read_nvfp4_rows MXFP4 branch (MIT, adapted).
    // MXFP4 -> NVFP4_KS repack (native W4A4): only on an explicit --nvfp4-precision w4a4 (ik also repacks when a valid input
    // scale exists, but an MXFP4 weight has none, so AUTO/INT8 leave MXFP4 alone there too). Needs no LLAMA_NVFP4_IMPORT_KS.
    const bool conv_nvfp4 = nvfp4_import_enabled();
    const bool conv_mxfp4 = nvfp4_precision == 1 /* W4A4 */;
    if (!conv_nvfp4 && !conv_mxfp4) {
        return false;
    }
    auto is_mxfp4 = [conv_mxfp4](const ggml_tensor * t) {
        return conv_mxfp4 && t->type == GGML_TYPE_MXFP4 && t->ne[0] % ggml_blck_size(GGML_TYPE_NVFP4_KS) == 0;
    };
    int n_import = 0;
    int n_mxfp4  = 0;
    for (const auto & w : weights) {
        n_import += conv_nvfp4 && w.tensor->type == GGML_TYPE_NVFP4;
        n_mxfp4  += is_mxfp4(w.tensor);
    }
    if (n_import + n_mxfp4 == 0) {
        return false;
    }

    ggml_context * ctx = ggml_init({ (n_import + n_mxfp4)*ggml_tensor_overhead(), nullptr, true });
    contexts.emplace_back(ctx);

    // This fork keeps weights_map (name -> llama_tensor_weight BY VALUE) beside `weights`, and get_weight()/
    // load_all_data() read the map, so every change made to `weights` is mirrored into the map (ik has only the vector).
    std::set<const ggml_tensor *> consumed;
    std::set<std::string>         consumed_names;
    int n_scale = 0;
    int n_input_scale = 0;
    int n_input_scale_kept = 0;
    // AUTO only trusts the checkpoint scales when it was not imatrix-quantized and is not marked prec_a4=false.
    bool auto_ok = true;
    {
        const int kp = gguf_find_key(meta, "general.tensor_extra.prec_a4");
        if (kp >= 0 && gguf_get_kv_type(meta, kp) == GGUF_TYPE_BOOL && !gguf_get_val_bool(meta, kp)) {
            auto_ok = false;
        }
        for (int i = 0; i < gguf_get_n_kv(meta) && auto_ok; ++i) {
            if (strncmp(gguf_get_key(meta, i), "quantize.imatrix.", 17) == 0) {
                auto_ok = false;
            }
        }
    }
    std::vector<ggml_tensor *> exps_unify; // ffn_up_exps / ffn_gate_exps share ONE scale (the max), as the fused up/gate kernel needs
    float exps_max = 0.0f;
    int n_repack = 0;
    for (auto & w : weights) {
        if (is_mxfp4(w.tensor)) {
            w.mxfp4_repack = true;
            const std::string mname(w.tensor->name);
            ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_NVFP4_KS, GGML_MAX_DIMS, w.tensor->ne);
            ggml_set_name(t, w.tensor->name);
            ggml_set_input_scale(t, -1.0f); // dynamic activation scale: native W4A4
            w.tensor = t;
            auto it_m = weights_map.find(mname);
            if (it_m != weights_map.end()) {
                it_m->second = w;
            }
            ++n_repack;
            continue;
        }
        if (!conv_nvfp4 || w.tensor->type != GGML_TYPE_NVFP4) {
            continue;
        }
        const std::string name(w.tensor->name);
        const std::string base = name.substr(0, name.rfind('.'));

        w.nvfp4_scale = { 1.0f };
        const auto * s = get_weight((base + ".scale").c_str());
        if (s && ggml_nelements(s->tensor) == s->tensor->ne[0] && (s->tensor->ne[0] == 1 || s->tensor->ne[0] == w.tensor->ne[2])) {
            const auto to_float = ggml_internal_get_type_traits(s->tensor->type).to_float;
            if (s->tensor->type != GGML_TYPE_F32 && !to_float) {
                throw std::runtime_error(format("%s: tensor '%s' has type %s, which cannot be read as a scale", __func__, ggml_get_name(s->tensor), ggml_type_name(s->tensor->type)));
            }
            w.nvfp4_scale.resize(s->tensor->ne[0]);
            std::vector<uint8_t> raw(ggml_nbytes(s->tensor));
            files.at(s->idx)->seek(s->offs, SEEK_SET);
            files.at(s->idx)->read_raw(raw.data(), raw.size());
            if (to_float) {
                to_float(raw.data(), w.nvfp4_scale.data(), w.nvfp4_scale.size());
            } else {
                memcpy(w.nvfp4_scale.data(), raw.data(), raw.size());
            }
            consumed.insert(s->tensor);
            consumed_names.insert(base + ".scale");
            ++n_scale;
        }
        // Activation (input) scale. SOURCE-READ: ik_llama.cpp b71fe5368 src/llama-model-loader.cpp import_nvfp4 (MIT, adapted).
        // W4A4: dynamic (-1). INT8: never W4A4 (0). AUTO: the checkpoint's .input_scale when valid, else 0.
        float in_scale = 0.0f;
        if (const auto * is = get_weight((base + ".input_scale").c_str())) {
            consumed.insert(is->tensor);
            consumed_names.insert(base + ".input_scale");
            ++n_input_scale;
            const auto to_float = ggml_internal_get_type_traits(is->tensor->type).to_float;
            if (ggml_nelements(is->tensor) == 1 && (is->tensor->type == GGML_TYPE_F32 || to_float)) {
                std::vector<uint8_t> raw(ggml_nbytes(is->tensor));
                files.at(is->idx)->seek(is->offs, SEEK_SET);
                files.at(is->idx)->read_raw(raw.data(), raw.size());
                float v = 0.0f;
                if (to_float) { to_float(raw.data(), &v, 1); } else { memcpy(&v, raw.data(), sizeof(v)); }
                if (std::isfinite(v) && v > 0.0f) {
                    in_scale = v;
                    ++n_input_scale_kept;
                }
            }
        }
        if (nvfp4_precision == 1 /* W4A4 */) {
            in_scale = -1.0f;
        } else if (nvfp4_precision == 2 /* INT8 */ || !auto_ok) {
            in_scale = 0.0f;
        }

        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_NVFP4_KS, GGML_MAX_DIMS, w.tensor->ne);
        ggml_set_name(t, w.tensor->name);
        ggml_set_input_scale(t, in_scale);
        if (in_scale > 0.0f && (name.find("ffn_up_exps") != std::string::npos || name.find("ffn_gate_exps") != std::string::npos)) {
            exps_unify.push_back(t);
            exps_max = std::max(exps_max, in_scale);
        }
        w.tensor = t;
        auto it_w = weights_map.find(name); // llama_tensor_weight has no default ctor: no operator[]
        if (it_w != weights_map.end()) {
            it_w->second = w;
        }
    }

    for (ggml_tensor * t : exps_unify) {
        ggml_set_input_scale(t, exps_max);
    }

    weights.erase(std::remove_if(weights.begin(), weights.end(), [&consumed](const llama_tensor_weight & w) {
        return consumed.count(w.tensor) > 0;
    }), weights.end());
    for (const std::string & n : consumed_names) {
        weights_map.erase(n);
    }

    if (n_import) {
        LLAMA_LOG_INFO("%s: converted %d %s tensors to %s; consumed %d .scale tensors; consumed %d .input_scale tensors, kept %d valid values (precision %d, auto_ok %d); mmap is off\n",
                __func__, n_import, ggml_type_name(GGML_TYPE_NVFP4), ggml_type_name(GGML_TYPE_NVFP4_KS), n_scale,
                n_input_scale, n_input_scale_kept, nvfp4_precision, (int) auto_ok);
    }
    if (n_repack) {
        LLAMA_LOG_INFO("%s: repacking %d %s tensors to %s for W4A4; mmap is off\n",
                __func__, n_repack, ggml_type_name(GGML_TYPE_MXFP4), ggml_type_name(GGML_TYPE_NVFP4_KS));
    }
    return true;
}

static void read_nvfp4_rows(const llama_model_loader::llama_tensor_weight & w, llama_file & file, const ggml_tensor * cur, uint8_t * dst, int r0, int nrows) {
    const int    ne1     = cur->ne[1];
    const int    n_scale = w.nvfp4_scale.size();
    const size_t row_in  = ggml_row_size(w.mxfp4_repack ? GGML_TYPE_MXFP4 : GGML_TYPE_NVFP4, cur->ne[0]);
    const size_t row_out = cur->nb[1];
    uint8_t * src = dst + nrows*(row_out - row_in);
    file.seek(w.offs + r0*row_in, SEEK_SET);
    file.read_raw(src, nrows*row_in);
    if (w.mxfp4_repack) {
        // The source rows sit at the tail of dst (row_out > row_in); row r's output ends at (r+1)*row_out, never past row r+1's
        // source start, so converting forward is safe. A scratch row keeps the converter's input and output apart.
        std::vector<uint8_t> row(row_in);
        int n_inexact = 0;
        for (int r = 0; r < nrows; ++r) {
            memcpy(row.data(), src + r*row_in, row_in);
            n_inexact += !ggml_mxfp4_to_nvfp4_ks(row.data(), dst + r*row_out, cur->ne[0]);
        }
        if (n_inexact) {
            LLAMA_LOG_WARN("%s: %s: %d of %d rows have block scales outside the UE4M3 range; those blocks were requantized\n",
                    __func__, ggml_get_name(cur), n_inexact, nrows);
        }
        return;
    }
    for (int r = 0; r < nrows; ++r) {
        memmove(dst + r*row_out + sizeof(float), src + r*row_in, row_in);
        memcpy(dst + r*row_out, &w.nvfp4_scale[((r0 + r)/ne1) % n_scale], sizeof(float));
    }
}

static void load_nvfp4_tensor(ggml_tensor * cur, size_t chunk_size, const llama_model_loader::llama_tensor_weight & w, llama_file & file,
        std::vector<no_init<uint8_t>> & buf, llama_model * model, std::mutex & mutex, bool check_tensors) {
    if (cur->buffer == nullptr) {
        std::lock_guard<std::mutex> lock(mutex);
        ggml_tensor * owner = cur->view_src ? cur->view_src : cur;
        if (owner->buffer == nullptr) {
            ggml_backend_buffer_t owned = ggml_backend_buft_alloc_buffer(llama_default_buffer_type_cpu(true), ggml_nbytes(owner));
            if (owned == nullptr) {
                throw std::runtime_error(format("unable to allocate buffer for tensor '%s'", ggml_get_name(owner)));
            }
            ggml_backend_tensor_alloc(owned, owner, ggml_backend_buffer_get_base(owned));
            ggml_backend_buffer_set_usage(owned, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            model->bufs.push_back(owned);
        }
        if (cur->view_src) {
            ggml_backend_view_init(cur);
        }
    }
    const bool host  = ggml_backend_buffer_is_host(cur->buffer);
    const int  nrows = ggml_nrows(cur);
    const int  chunk = host ? nrows : std::max<int>(1, chunk_size/cur->nb[1]);
    for (int r0 = 0; r0 < nrows; r0 += chunk) {
        const int    nr   = std::min(chunk, nrows - r0);
        const size_t size = nr*cur->nb[1];
        if (!host) {
            buf.resize(size);
        }
        uint8_t * dst = host ? (uint8_t *) cur->data : (uint8_t *) buf.data();
        read_nvfp4_rows(w, file, cur, dst, r0, nr);
        if (check_tensors && !ggml_validate_row_data(cur->type, dst, size)) {
            throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
        }
        if (!host) {
            std::lock_guard<std::mutex> lock(mutex);
            ggml_backend_tensor_set(cur, dst, r0*cur->nb[1], size);
        }
    }
}

// for backwards compatibility, does not support ggml-backend
void llama_model_loader::load_data_for(struct ggml_tensor * cur) const {
    // UNIT 1: metadata-only source -- bytes come from the caller's callback, not from `files`
    // (there may be none). Checked FIRST and returns, so every pre-existing caller (set_tensor_data
    // is null unless the new metadata-only ctor set it) falls through to the unchanged code below.
    if (set_tensor_data) {
        GGML_ASSERT(cur->data != nullptr);
        set_tensor_data(cur, set_tensor_data_ud);
        if (check_tensors && !ggml_validate_row_data(cur->type, cur->data, ggml_nbytes(cur))) {
            throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
        }
        return;
    }

    const auto & w = require_weight(ggml_get_name(cur));

    if (!w.nvfp4_scale.empty() || w.mxfp4_repack) {
        read_nvfp4_rows(w, *files.at(w.idx), cur, (uint8_t *) cur->data, 0, ggml_nrows(cur));
    } else if (use_mmap) {
        const auto & mapping = mappings.at(w.idx);
        if (cur->data == nullptr) {
            cur->data = (uint8_t *)mapping->addr() + w.offs;
        } else {
            memcpy(cur->data, (uint8_t *)mapping->addr() + w.offs, ggml_nbytes(cur));
        }
    } else {
        GGML_ASSERT(cur->data != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(cur->data, ggml_nbytes(cur));
    }

    if (check_tensors && !ggml_validate_row_data(cur->type, cur->data, ggml_nbytes(cur))) {
        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
    }
}

// upstream (merge 2026-11-04, MERGE_HEAD src/llama-model-loader.cpp:1540): read a byte range of a weight
const void * llama_model_loader::load_data_range(const llama_tensor_weight & w, size_t offs, size_t size, void * buf) const {
    GGML_ASSERT(offs + size <= ggml_nbytes(w.tensor));

    const void * data = buf;

    if (use_mmap) {
        data = (const uint8_t *) mappings.at(w.idx)->addr() + w.offs + offs;
    } else {
        GGML_ASSERT(buf != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs + offs, SEEK_SET);
        file->read_raw(buf, size);
    }

    if (check_tensors && !ggml_validate_row_data(w.tensor->type, data, size)) {
        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(w.tensor)));
    }

    return data;
}

// Returns false if canceled by progress_callback
bool llama_model_loader::load_all_data(
            struct ggml_context * ctx,
            [[maybe_unused]] llama_model * model,
            llama_buf_map & bufs_mmap,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data) {
    GGML_ASSERT(size_data != 0 && "call init_mappings() first");

    std::vector<std::future<std::pair<ggml_tensor *, bool>>> validation_result;

    // Number of worker threads for cuda and host tensor loading.
    const int n_workers = 8;

    std::vector<std::vector<no_init<uint8_t>>> read_bufs(n_workers);

    // One pinned staging buffer per worker for async uploads (any backend, see upload_backend below)
    constexpr size_t buffer_size = 16 * 1024 * 1024; // 16MB

    std::vector<ggml_backend_buffer_t> host_buffers;
    std::vector<void*> host_ptrs;
    std::vector<ggml_backend_event_t> events;

#if defined(GGML_USE_CUDA)
#if !defined(_WIN32)
    std::vector<std::unique_ptr<llama_mmap>> split_mappings(files.size());
#endif
    // --split-mode graph reads each tensor whole into host memory, on every worker at once. Bound the bytes in flight by
    // what the OS can allocate (src/llama-load-budget.h); UNLIMITED until the first such tensor probes the OS once.
    llama_load_budget::semaphore split_budget;
    std::once_flag               split_budget_once;

#endif

    // Async uploads from pinned host memory, for any device that can do them (upstream: generic probe on dev props caps.async +
    // caps.host_buffer + caps.events; this fork had CUDA only). CUDA keeps its own init (it takes `model`); every other backend goes
    // through the capability probe.
    // SOURCE-READ: upstream/unsloth-llama.cpp/src/llama-model-loader.cpp:1594-1666
    // SOURCE-NUANCE: upstream requires buft == the device's DEFAULT buffer type (a split/host/extra buft is refused), allocates
    // every pinned buffer before the backend is initialised and returns null on the first failure; here a failure also frees what
    // was already allocated, and tensors outside the probed buffer type never take the generic async path.
    ggml_backend_t                upload_backend = nullptr;
    ggml_backend_buffer_type_t    upload_buft    = nullptr;
    bool                          upload_is_cuda = false;
    if (!use_mmap && !check_tensors) {
        // PR #152: one file index can now own several buffers; the async-upload probe only
        // needs any one of them to ask the backend about its capabilities.
        ggml_backend_buffer_t buf = bufs_mmap.count(0) && !bufs_mmap.at(0).empty() ? bufs_mmap.at(0).front() : nullptr;
        ggml_backend_buffer_type_t buffer_type = buf ? ggml_backend_buffer_get_type(buf) : nullptr;
        ggml_backend_buffer_type_t host_buft = nullptr;
#if defined(GGML_USE_CUDA)
        if (buf) {
            for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
                auto * cuda_buffer_type = ggml_backend_cuda_buffer_type(i);
                if (buffer_type == cuda_buffer_type) {
                    upload_backend = ggml_backend_cuda_init(i, nullptr, model);
                    upload_is_cuda = upload_backend != nullptr;
                    host_buft      = llama_default_buffer_type_cpu(true);
                    break;
                }
            }
        }
#endif
        if (buf && !upload_backend) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buffer_type);
            if (dev && buffer_type == ggml_backend_dev_buffer_type(dev)) {
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                ggml_backend_buffer_type_t dev_host_buft = ggml_backend_dev_host_buffer_type(dev);
                if (props.caps.async && props.caps.host_buffer && props.caps.events && dev_host_buft) {
                    upload_backend = ggml_backend_dev_init(dev, nullptr);
                    host_buft      = dev_host_buft;
                }
            }
        }

        // create pinned memory buffers and events for synchronization.
        if (upload_backend) {
            upload_buft = buffer_type;
            bool ok = true;
            for (size_t idx = 0; idx < (size_t)n_workers; ++idx) {
                ggml_backend_buffer_t hb = ggml_backend_buft_alloc_buffer(host_buft, buffer_size);
                // Ported from upstream 2026-09-08: ggml_backend_event_new now takes a device
                // handle, not a backend -- this fork's ggml-backend.h already carries the newer
                // signature (ggml-backend.h:124), only this call site was still on the old one.
                ggml_backend_event_t ev = ggml_backend_event_new(ggml_backend_get_device(upload_backend));
                if (!hb || !ev) {
                    if (hb) { ggml_backend_buffer_free(hb); }
                    if (ev) { ggml_backend_event_free(ev); }
                    ok = false;
                    break;
                }
                host_buffers.emplace_back(hb);
                host_ptrs.emplace_back(ggml_backend_buffer_get_base(hb));
                events.emplace_back(ev);
            }
            if (ok) {
                // Force creation of the upload stream now
                ggml_backend_event_record(events[0], upload_backend);
                ggml_backend_event_synchronize(events[0]);
                if (!upload_is_cuda) {
                    LLAMA_LOG_INFO("%s: async uploads enabled for device %s\n", __func__,
                            ggml_backend_dev_name(ggml_backend_get_device(upload_backend)));
                }
            } else {
                for (auto * e : events) { ggml_backend_event_free(e); }
                for (auto * h : host_buffers) { ggml_backend_buffer_free(h); }
                events.clear(); host_buffers.clear(); host_ptrs.clear();
                ggml_backend_free(upload_backend);
                upload_backend = nullptr;
                LLAMA_LOG_WARN("%s: could not allocate pinned staging buffers for async uploads, using synchronous uploads\n", __func__);
            }
        }
    }


    // Tensors are loaded in a threadpool. A lot of the ops are serialized
    // and this is the mutex we use.
    std::mutex load_mutex;
    // A file built from a FILE* (llama_model_load_from_file_ptr) has no path to reopen, so it cannot be cloned per
    // tensor; its tensors are read through the one shared handle, seek+read under this mutex.
    std::mutex file_ptr_mutex;

    // Load model weights into a backing buffer:
    // * mmap:               host page cache        Serial
    // * host:               system ram             Parallel
    // * cuda:               GPU                    Parallel
    // * --split-mode graph: GPU                    Parallel
    // * rest:               All other backends,    Serial
    auto load_tensor = [&](ggml_tensor * cur, int thread_idx) -> size_t {
        // UNIT 1: metadata-only source -- same callback substitution as load_data_for() above,
        // checked first so every pre-existing (file-backed) caller is unaffected.
        if (set_tensor_data) {
            const size_t n_size = ggml_nbytes(cur);
            GGML_ASSERT(cur->data != nullptr);
            set_tensor_data(cur, set_tensor_data_ud);
            if (check_tensors && !ggml_validate_row_data(cur->type, cur->data, n_size)) {
                throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
            }
            return n_size;
        }

        const auto * weight = get_weight(ggml_get_name(cur));
        GGML_ASSERT(weight != nullptr);
        GGML_ASSERT(weight->idx < files.size());
        const size_t n_size = ggml_nbytes(cur);
        const bool shared_fp = files.at(weight->idx)->get_path() == "(file*)";
        std::unique_ptr<llama_file> file_clone = shared_fp ? nullptr : files.at(weight->idx)->clone();
        llama_file * file = shared_fp ? files.at(weight->idx).get() : file_clone.get();
        std::unique_lock<std::mutex> fp_lock(file_ptr_mutex, std::defer_lock);

        // Ollama qwen35 MTP compat (apply_ollama_qwen35_mtp_compat): a few small F32 norms are stored without llama.cpp's
        // +1.0. Read, shift, then store. Only these tensors leave the file mapping; every other tensor of the load stays mapped.
        //  - cur->buffer set (--no-mmap, -rtr, a device buffer under -ngl, any context whose mmap buffer path is off): the tensor
        //    already has writable storage, upload into it.
        //  - cur->buffer null (mmap load, host context: llama.cpp creates CPU_Mapped buffers over the file and binds tensors here):
        //    binding it would alias the read-only mapping, so it gets owned storage instead, the same way as the repackables
        //    under -rtr / dense tensors under --defer-ple below (owned CPU buffer, kept alive by model->bufs).
        if (!compat_norm_add_one.empty() &&
            std::find(compat_norm_add_one.begin(), compat_norm_add_one.end(), ggml_get_name(cur)) != compat_norm_add_one.end()) {
            GGML_ASSERT(cur->type == GGML_TYPE_F32 && cur->view_src == nullptr);
            std::vector<float> buf(n_size / sizeof(float));
            if (shared_fp) fp_lock.lock();
            file->seek(weight->offs, SEEK_SET);
            file->read_raw(buf.data(), n_size);
            if (shared_fp) fp_lock.unlock();
            for (float & x : buf) {
                x += 1.0f;
            }
            if (check_tensors && !ggml_validate_row_data(cur->type, buf.data(), n_size)) {
                throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
            }
            if (cur->buffer != nullptr) {
                ggml_backend_tensor_set(cur, buf.data(), 0, n_size);
                return n_size;
            }
            GGML_ASSERT(use_mmap && model != nullptr && cur->data == nullptr);
            std::lock_guard<std::mutex> lock(load_mutex);   // model->bufs / mlock_bufs are shared with the other workers
            ggml_backend_buffer_t owned = ggml_backend_buft_alloc_buffer(llama_default_buffer_type_cpu(true), n_size);
            if (owned == nullptr) {
                throw std::runtime_error(format("unable to allocate compat buffer for tensor '%s'", ggml_get_name(cur)));
            }
            uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(owned);
            memcpy(base, buf.data(), n_size);
            ggml_backend_tensor_alloc(owned, cur, base);
            ggml_backend_buffer_set_usage(owned, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            model->bufs.push_back(owned);
            if (lmlocks) {
                model->mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = model->mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(owned));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(owned));
            }
            return n_size;
        }

        if (!weight->nvfp4_scale.empty() || weight->mxfp4_repack) {
            load_nvfp4_tensor(cur, buffer_size, *weight, *file, read_bufs[thread_idx], model, load_mutex, check_tensors);
            return n_size;
        }

        // mmap. Serialized.
        if (use_mmap) {
            // Stream device tensors from disk: Windows has no per-range discard for READONLY views,
            // and on integrated GPUs the upload from the mapping is slow.
#if defined(_WIN32)
            const bool stream_from_file = cur->buffer != nullptr && !ggml_backend_buffer_is_host(cur->buffer);
            std::lock_guard<std::mutex> lock(load_mutex);
#else
            const bool stream_from_file = cur->buffer != nullptr &&
                std::find(integrated_bufts.begin(), integrated_bufts.end(), ggml_backend_buffer_get_type(cur->buffer)) != integrated_bufts.end();
            // Integrated GPU tensors load without the lock. Each worker has its own file and read buffer, and the copy uses cudaStreamPerThread.
            std::unique_lock<std::mutex> lock(load_mutex, std::defer_lock);
            if (!stream_from_file) {
                lock.lock();
            }
#endif
            // Owned buffers streamed from the file: repackables under -rtr, everything
            // not deferred under explicit --no-mmap; deferred tables stay aliased.
            if (defer_ple && !ple_tensor_index.empty() && lmlocks == nullptr &&
                    cur->buffer == nullptr && cur->view_src == nullptr && n_size > 0 &&
                    !ple_range_overlaps(weight->idx, weight->offs, weight->offs + n_size) &&
                    (defer_copy_dense || (repack_tensors && (ggml_type) iqk_repacked_type(cur) != cur->type))) {
                bool has_views = false;
                for (auto * v = ggml_get_first_tensor(ctx); v != NULL; v = ggml_get_next_tensor(ctx, v)) {
                    if (v->view_src == cur) {
                        has_views = true;
                        break;
                    }
                }
                if (!has_views) {
                    ggml_backend_buffer_t owned = ggml_backend_buft_alloc_buffer(llama_default_buffer_type_cpu(true), n_size);
                    if (owned == nullptr) {
                        throw std::runtime_error(format("unable to allocate repack buffer for tensor '%s'", ggml_get_name(cur)));
                    }
                    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(owned);
                    file->seek(weight->offs, SEEK_SET);
                    file->read_raw(base, n_size);
                    if (check_tensors && !ggml_validate_row_data(cur->type, base, n_size)) {
                        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                    }
                    ggml_backend_tensor_alloc(owned, cur, base);
                    ggml_backend_buffer_set_usage(owned, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    model->bufs.push_back(owned);
                    return n_size;
                }
            }
            if (stream_from_file) {
                auto & read_buf = read_bufs[thread_idx];
                read_buf.resize(n_size);
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(read_buf.data(), n_size);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                    throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                }
                return n_size;
            }
            const auto & mapping = mappings.at(weight->idx);
            ggml_backend_buffer_t buf_mmap = nullptr;
            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;

            // PR #152: a file index can map to several buffers now, so pick the one that
            // actually CONTAINS this tensor rather than assuming there is only one.
            if (bufs_mmap.count(weight->idx)) {
                for (ggml_backend_buffer_t b : bufs_mmap.at(weight->idx)) {
                    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(b);
                    if (data >= base && data + n_size <= base + ggml_backend_buffer_get_size(b)) {
                        buf_mmap = b;
                        break;
                    }
                }
            }

            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                            return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                            }));
            }

            // Merged views share no file range: malloc the base once if needed,
            // resolve the view, stream this slice without faulting the mapping on Windows.
            if (cur->view_src != NULL && cur->buffer == nullptr) {
                struct ggml_tensor * base = cur->view_src;
                if (base->buffer == nullptr) {
                    const size_t base_size = ggml_nbytes(base);
                    ggml_backend_buffer_t bbuf = ggml_backend_buft_alloc_buffer(llama_default_buffer_type_cpu(true), base_size);
                    if (bbuf == nullptr) {
                        throw std::runtime_error(format("unable to allocate merged base for tensor '%s'", ggml_get_name(cur)));
                    }
                    ggml_backend_tensor_alloc(bbuf, base, ggml_backend_buffer_get_base(bbuf));
                    ggml_backend_buffer_set_usage(bbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    model->bufs.push_back(bbuf);
                    if (lmlocks) {
                        model->mlock_bufs.emplace_back(new llama_mlock);
                        auto & mlock_buf = model->mlock_bufs.back();
                        mlock_buf->init   (ggml_backend_buffer_get_base(bbuf));
                        mlock_buf->grow_to(ggml_backend_buffer_get_size(bbuf));
                    }
                }
                ggml_backend_view_init(cur);
#if defined(_WIN32)
                // File stream, not mapping: slice copies would pin tens of GB
                // of fused-expert file pages in the working set for no benefit.
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(cur->data, n_size);
#else
                ggml_backend_tensor_set(cur, data, 0, n_size);
#endif
                return n_size;
            }

            GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
            if (buf_mmap && cur->data == nullptr) {
                ggml_backend_tensor_alloc(buf_mmap, cur, data);

                if (lmlocks) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + n_size);
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
            return n_size;
        }

        // host. Parallel.
        if (ggml_backend_buffer_is_host(cur->buffer)) {
            if (shared_fp) fp_lock.lock();
            file->seek(weight->offs, SEEK_SET);
            file->read_raw(cur->data, n_size);
            if (shared_fp) fp_lock.unlock();
            if (check_tensors) {
                std::lock_guard<std::mutex> lock(load_mutex);
                validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                            return std::make_pair(cur, ggml_validate_row_data(cur->type, cur->data, n_size));
                            }));
            }
            return n_size;
        }

        // device with async uploads (CUDA, or any backend the probe above accepted). Parallel
        // If upload_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the device.
        // CUDA keeps its historic behaviour (every tensor that gets here); another backend only takes tensors of the buffer
        // type it was probed for.
        if (upload_backend && (upload_is_cuda || (cur->buffer && ggml_backend_buffer_get_type(cur->buffer) == upload_buft))) {
            if (shared_fp) fp_lock.lock();   // held for the whole chunked read: it moves the shared handle's position
            file->seek(weight->offs, SEEK_SET);

            size_t bytes_read = 0;

            while (bytes_read < n_size) {
                size_t read_iteration = std::min<size_t>(buffer_size, n_size - bytes_read);

                ggml_backend_event_synchronize(events[thread_idx]);
                file->read_raw(host_ptrs[thread_idx], read_iteration);
                ggml_backend_tensor_set_async(upload_backend, cur, host_ptrs[thread_idx], bytes_read, read_iteration);
                // Same ggml_backend_event_record(event, backend) signature fix as above.
                ggml_backend_event_record(events[thread_idx], upload_backend);

                bytes_read += read_iteration;
            }
            return n_size;
        }

#if defined(GGML_USE_CUDA)
        // --split-mode graph. Parallel
        const char * buffer_name = ggml_backend_buffer_name(cur->buffer);
        const bool   is_probably_split_mode_graph = std::strncmp(buffer_name, GGML_CUDA_NAME, strlen(GGML_CUDA_NAME)) == 0;
        if (is_probably_split_mode_graph) {
            std::call_once(split_budget_once, [&] {
                const auto     avail  = llama_load_budget::host_allocatable_bytes();
                const uint64_t budget = llama_load_budget::budget_bytes(avail);
                split_budget.set_budget(budget);
                if (avail) {
                    LLAMA_LOG_INFO("%s: -sm graph load budget %.1f MiB (2/3 of %.1f MiB allocatable host RAM); "
                                   "a tensor larger than the budget loads alone\n", __func__,
                                   budget / (1024.0 * 1024.0), *avail / (1024.0 * 1024.0));
                } else {
                    LLAMA_LOG_INFO("%s: -sm graph load budget: host RAM probe gave no answer, parallel loads unbounded\n", __func__);
                }
            });
            llama_load_budget::semaphore::guard budget_guard(split_budget, n_size);
#if !defined(_WIN32)
            llama_mmap * mapping;
            {
                std::lock_guard<std::mutex> lock(load_mutex);
                auto & m = split_mappings[weight->idx];
                if (!m) {
                    m.reset(new llama_mmap(files.at(weight->idx).get(), 0, ggml_is_numa()));
                }
                mapping = m.get();
            }
            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;
            ggml_backend_tensor_set(cur, data, 0, n_size);
            if (check_tensors && !ggml_validate_row_data(cur->type, data, n_size)) {
                throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
            }
            mapping->dontneed_fragment(weight->offs, weight->offs + n_size);
            return n_size;
#else
            // A local, not read_bufs[thread_idx]: it is freed before budget_guard hands its bytes back, so memory a
            // worker still holds is never counted as free by the next one.
            std::vector<no_init<uint8_t>> read_buf(n_size);
            if (shared_fp) fp_lock.lock();
            file->seek(weight->offs, SEEK_SET);
            file->read_raw(read_buf.data(), n_size);
            ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
            if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
            }
            return n_size;
#endif
        }
#endif
        // rest. Serialized.
        {
            std::lock_guard<std::mutex> lock(load_mutex);
            auto & read_buf = read_bufs[thread_idx];
            read_buf.resize(n_size);
            if (shared_fp) fp_lock.lock();
            file->seek(weight->offs, SEEK_SET);
            file->read_raw(read_buf.data(), n_size);
            ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
            if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
            }
            return n_size;
        }
    };

    // An iterator that the threadpool shares to get the next tensor for loading.
    // Each thread loops over this function until there are no more tensors left to load.
    auto make_next_tensor = [&]() {
        // Visit order. Without mmap, a tensor in a non-host buffer is staged through a host buffer sized like the tensor (the
        // "rest" path's read_bufs[thread_idx].resize(n_size)); upstream loads those biggest-first so each staging buffer is
        // allocated once, at its final size, while the fewest weights are resident, instead of growing through realloc steps.
        // SOURCE-READ: upstream/unsloth-llama.cpp/src/llama-model-loader.cpp:1680-1691
        // SOURCE-NUANCE: stable sort, staged (non-host) tensors first then size descending; host tensors keep context order after
        // them; the staged test reads a->buffer, which is null under mmap, so upstream gates the sort on !use_mmap. Here the sort is
        // also skipped for metadata-only loads (set_tensor_data), and the order is shared by 8 workers through one atomic index.
        auto order = std::make_shared<std::vector<ggml_tensor *>>();
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            order->push_back(t);
        }
        if (!use_mmap && !set_tensor_data) {
            std::stable_sort(order->begin(), order->end(), [](const ggml_tensor * a, const ggml_tensor * b) {
                const bool staged_a = a->buffer && !ggml_backend_buffer_is_host(a->buffer);
                const bool staged_b = b->buffer && !ggml_backend_buffer_is_host(b->buffer);
                if (staged_a != staged_b) {
                    return staged_a;
                }
                return staged_a && ggml_nbytes(a) > ggml_nbytes(b);
            });
        }
        auto cursor = std::make_shared<std::atomic<size_t>>(0);
        auto next_tensor = [this, order, cursor]() -> ggml_tensor * {
            while (true) {
                const size_t i = cursor->fetch_add(1);
                if (i >= order->size()) {
                    return nullptr;
                }
                ggml_tensor * cur = (*order)[i];

                // With split experts models get_weight can return nullptr. A metadata-only loader (set_tensor_data) has NO weights
                // at all: every tensor's bytes come from the callback, so every tensor must be handed to load_tensor(). Skipping on
                // get_weight() == nullptr left every tensor in a GPU buffer unfilled (MEASURED 2026-10-04: compute-sanitizer initcheck
                // flagged fused_rms_norm_f32 reading uninitialised device memory; all device logits NaN on test-llama-archs -a llama).
                if (set_tensor_data || get_weight(ggml_get_name(cur))) {
                    return cur;
                }
            }
        };
        return next_tensor;
    };
    auto next_tensor = make_next_tensor();

    std::atomic<size_t>  loaded{size_done}; // bytes loaded so far, for the progress bar
    std::atomic<bool>    cancelled{false};
    std::atomic<bool>    failed{false};
    std::exception_ptr   first_exception;

    // threadpool worker.
    auto worker = [&](int thread_idx) {
        try {
            while (!cancelled.load() && !failed.load()) {
                if (progress_callback) {
                    const size_t done   = loaded.load(std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(load_mutex);
                    if (!progress_callback((float) done / size_data, progress_callback_user_data)) {
                        cancelled.store(true);
                        break;
                    }
                }
                ggml_tensor * cur = next_tensor();
                if (!cur) {
                    break;
                }
                const size_t n_size = load_tensor(cur, thread_idx);
                loaded.fetch_add(n_size, std::memory_order_relaxed);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(load_mutex);
            if (!failed.exchange(true)) {
                first_exception = std::current_exception();
            }
        }
    };

    std::vector<std::thread> pool;
    for (int thread_idx = 0; thread_idx < n_workers; ++thread_idx) {
        pool.emplace_back(worker, thread_idx);
    }
    for (auto & t : pool) {
        t.join();
    }

    size_done = loaded.load();
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
    if (cancelled.load()) {
        return false;
    }

    // free temporary resources used for async uploads
    if (upload_backend) {
        for (size_t idx = 0; idx < events.size(); ++idx) {
            ggml_backend_event_synchronize(events[idx]);
            ggml_backend_event_free(events[idx]);
            ggml_backend_buffer_free(host_buffers[idx]);
        }
        ggml_backend_free(upload_backend);
    }

    // check validation results
    bool validation_failed = false;
    for (auto & future : validation_result) {
        auto result = future.get();
        if (!result.second) {
            LLAMA_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, ggml_get_name(result.first));
            validation_failed = true;
        }
    }
    if (validation_failed) {
        throw std::runtime_error("found tensors with invalid data");
    }

    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto & mapping = mappings.at(idx);
                mapping->unmap_fragment(0, mmap_used.first);
                if (mmap_used.second != 0) {
                    mapping->unmap_fragment(mmap_used.second, mapping->size());
                }
            }
        }
        if (progress_callback) {
            // Even though the model is done loading, we still honor
            // cancellation since we need to free allocations.
            return progress_callback(1.0f, progress_callback_user_data);
        }
    }

    return true;
}

template<>
bool llama_model_loader::get_key(const enum llm_kv kid, enum llama_pooling_type & result, const bool required) {
    uint32_t tmp;
    const bool found = get_key(kid, tmp, required);
    if (found) {
        result = (enum llama_pooling_type) tmp;
    } else {
        result = LLAMA_POOLING_TYPE_UNSPECIFIED;
    }
    return found;
}
template bool llama_model_loader::get_key<bool>       (enum llm_kv kid, bool & result,        bool required);
template bool llama_model_loader::get_key<float>      (enum llm_kv kid, float & result,       bool required);
template bool llama_model_loader::get_key<uint32_t>   (enum llm_kv kid, uint32_t & result,    bool required);
template bool llama_model_loader::get_key<std::string>(enum llm_kv kid, std::string & result, bool required);

template bool llama_model_loader::get_key_or_arr<std::array<int, 4>>(enum llm_kv kid, std::array<int, 4> & result, uint32_t n, bool required, bool allow_partial);
template bool llama_model_loader::get_key_or_arr<std::array<uint32_t, 512>>(enum llm_kv kid, std::array<uint32_t, 512> & result, uint32_t n, bool required, bool allow_partial);
template bool llama_model_loader::get_key_or_arr<std::array<float, 512>>(enum llm_kv kid, std::array<float, 512> & result, uint32_t n, bool required, bool allow_partial);
// referenced by src/models classes (minimax-01 reads a per-layer bool pattern) that became reachable with llama_model_create_for_arch; undefined at link until instantiated here
template bool llama_model_loader::get_key_or_arr<std::array<bool, 512>>(enum llm_kv kid, std::array<bool, 512> & result, uint32_t n, bool required, bool allow_partial);

template std::enable_if<std::is_integral<unsigned int>::value, bool>::type llama_model_loader::get_arr_n<unsigned int>(const std::string &, unsigned int &, bool);
template std::enable_if<std::is_integral<unsigned int>::value, bool>::type llama_model_loader::get_arr_n<unsigned int>(enum llm_kv, unsigned int&, bool);
template bool llama_model_loader::get_arr<uint32_t>(const std::string &, std::vector<uint32_t> &, bool);
template bool llama_model_loader::get_arr<uint64_t>(const std::string &, std::vector<uint64_t> &, bool);
template bool llama_model_loader::get_arr<int32_t, 8>(const std::string &, std::array<int32_t, 8> &, bool);
template bool llama_model_loader::get_arr<uint32_t, 8>(const std::string &, std::array<uint32_t, 8> &, bool);
// per-layer arrays read by dots3note, granite-swa and spark2-5 (src/models classes reachable through llama_model_create_for_arch)
template bool llama_model_loader::get_arr<std::array<uint32_t, 512>>(enum llm_kv, std::array<uint32_t, 512> &, bool);
template bool llama_model_loader::get_arr<std::array<int, 512>>(enum llm_kv, std::array<int, 512> &, bool);
// std::vector<int32_t> keyed by llm_kv: dflash/eagle3 (target_layer_ids, llama-model.h) and granite-switch (adapter token ids, std::vector<llama_token>).
// Found by a clean Linux GCC 13 link (libllama.so: undefined reference to get_arr<std::vector<int>>(llm_kv, ...), 76+54 hits).
template bool llama_model_loader::get_arr<std::vector<int32_t>>(enum llm_kv, std::vector<int32_t> &, bool);
template bool llama_model_loader::get_arr<std::vector<std::string>>(enum llm_kv, std::vector<std::string> &, bool);
// Instantiations a Linux shared libllama.so link reported missing (WSL2 gcc 15, hidden visibility); MSVC/static links resolve them from the header-visible uses.
template bool llama_model_loader::get_arr<std::vector<uint32_t>>(enum llm_kv, std::vector<uint32_t> &, bool);
template bool llama_model_loader::get_arr<std::vector<uint64_t>>(enum llm_kv, std::vector<uint64_t> &, bool);
template bool llama_model_loader::get_arr<std::array<uint64_t, 64>>(enum llm_kv, std::array<uint64_t, 64> &, bool);
template bool llama_model_loader::get_arr<std::array<uint64_t, 8>>(enum llm_kv, std::array<uint64_t, 8> &, bool);
template bool llama_model_loader::get_key<int>(enum llm_kv kid, int & result, bool required);

// adopted from upstream in this merge: both trivial, safe, use only already-existing fields.
std::string llama_model_loader::ftype_name() const {
    return llama_ftype_name(ftype);
}

void llama_model_loader::print_info() const {
    LLAMA_LOG_INFO("%s: file format = %s\n", __func__, llama_file_version_name(fver));
    LLAMA_LOG_INFO("%s: file type   = %s\n", __func__, llama_ftype_name(ftype));
    if (n_bytes < GiB) {
        LLAMA_LOG_INFO("%s: file size   = %.2f MiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0,        n_bytes*8.0/n_elements);
    } else {
        LLAMA_LOG_INFO("%s: file size   = %.2f GiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0/1024.0, n_bytes*8.0/n_elements);
    }
}

// Prism ternary Hadamard rotations (weights are stored pre-rotated in the GGUF), ik_llama.cpp #2550 (MIT)
void llm_load_hadamard(llama_model_loader & ml, llama_model & model) {
    if (getenv("IK_HADAMARD_DISABLE")) {
        LLAMA_LOG_WARN("%s: disabled via IK_HADAMARD_DISABLE\n", __func__);
        return;
    }
    uint32_t version = 0;
    if (!ml.get_key("prism.hadamard.version", version, false)) {
        return;
    }
    if (version != 1 && version != 2) {
        LLAMA_LOG_WARN("%s: unsupported prism.hadamard.version %u, ignoring Hadamard rotations\n", __func__, version);
        return;
    }

    uint32_t block_size = 0;
    if (!ml.get_key("prism.hadamard.block_size", block_size, false) || block_size == 0 || (block_size & (block_size - 1)) != 0) {
        LLAMA_LOG_WARN("%s: invalid prism.hadamard.block_size, ignoring Hadamard rotations\n", __func__);
        return;
    }

    const auto get_str_arr = [&](const std::string & key) {
        std::vector<std::string> out;
        const int kid = gguf_find_key(ml.meta, key.c_str());
        if (kid < 0 || gguf_get_kv_type(ml.meta, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ml.meta, kid) != GGUF_TYPE_STRING) {
            return out;
        }
        const size_t n = gguf_get_arr_n(ml.meta, kid);
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            out.emplace_back(gguf_get_arr_str(ml.meta, kid, i));
        }
        return out;
    };

    const std::vector<std::string> weight_names  = get_str_arr("prism.hadamard.weight_names");
    const std::vector<std::string> inverse_names = get_str_arr("prism.hadamard.inverse_weight_names");

    if (weight_names.empty() && inverse_names.empty()) {
        return;
    }

    std::vector<int32_t> sign_widths;
    std::vector<int32_t> sign_values;
    ml.get_arr("prism.hadamard.sign_widths", sign_widths, false);
    ml.get_arr("prism.hadamard.sign_values", sign_values, false);

    std::map<int64_t, std::vector<float>> signs_by_width;
    {
        size_t off = 0;
        for (const int32_t width : sign_widths) {
            if (width <= 0 || width % (int64_t) block_size != 0 || off + (size_t) width > sign_values.size()) {
                LLAMA_LOG_WARN("%s: invalid prism.hadamard sign width %d, ignoring Hadamard rotations\n", __func__, width);
                return;
            }
            std::vector<float> signs(width);
            for (int64_t j = 0; j < width; ++j) {
                const int32_t v = sign_values[off + j];
                if (v != 1 && v != -1) {
                    LLAMA_LOG_WARN("%s: prism.hadamard sign values must be +/-1, ignoring Hadamard rotations\n", __func__);
                    return;
                }
                signs[j] = (float) v;
            }
            signs_by_width[width] = std::move(signs);
            off += width;
        }
    }

    bool gdn_v_grouped = false;
    ml.get_key("prism.hadamard.gdn_v_grouped", gdn_v_grouped, false);

    std::set<std::string> rotated_names(weight_names.begin(), weight_names.end());
    rotated_names.insert(inverse_names.begin(), inverse_names.end());

    // Signs are created on the same buffer type (device) as the weight.
    std::map<std::string, ggml_backend_buffer_type_t> weight_buft;
    for (const auto & kv : model.tensors_by_name) {
        if (kv.second == nullptr || kv.second->buffer == nullptr || rotated_names.find(kv.first) == rotated_names.end()) {
            continue;
        }
        ggml_backend_buffer_type_t wb = ggml_backend_buffer_get_type(kv.second->buffer);
        weight_buft[kv.first] = ggml_backend_buft_is_host(wb) ? llama_default_buffer_type_cpu(true) : wb;
    }

    // One signs tensor per (width, device).
    std::map<std::pair<int64_t, ggml_backend_buffer_type_t>, ggml_tensor *> sign_tensors;

    const auto make_signs = [&](int64_t width, ggml_backend_buffer_type_t buft) -> ggml_tensor * {
        const auto key = std::make_pair(width, buft);
        const auto it = sign_tensors.find(key);
        if (it != sign_tensors.end()) {
            return it->second;
        }
        const auto sd = signs_by_width.find(width);
        if (sd == signs_by_width.end()) {
            return nullptr;
        }
        ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            return nullptr;
        }
        ggml_tensor * signs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        ggml_set_name(signs, ("prism.hadamard.signs." + std::to_string(width)).c_str());
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf == nullptr) {
            ggml_free(ctx);
            return nullptr;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(signs, sd->second.data(), 0, sd->second.size() * sizeof(float));

        model.ctxs.push_back(ctx);
        model.bufs.push_back(buf);
        sign_tensors[key] = signs;
        return signs;
    };

    const auto fill_transform = [&](const std::string & name) {
        llama_hadamard_transform & t = model.hadamard_map[name];
        const ggml_tensor * meta = ml.get_tensor_meta(name.c_str());
        if (meta == nullptr) {
            LLAMA_LOG_WARN("%s: prism.hadamard weight not found: %s\n", __func__, name.c_str());
            return;
        }
        const int64_t width = meta->ne[0];
        if (width <= 0 || width % (int64_t) block_size != 0) {
            LLAMA_LOG_WARN("%s: prism.hadamard weight %s has ne0=%lld, not a multiple of block %u\n",
                    __func__, name.c_str(), (long long) width, block_size);
            return;
        }
        t.block_size = block_size;
        if (!signs_by_width.empty()) {
            ggml_backend_buffer_type_t buft = llama_default_buffer_type_cpu(true);
            if (const auto wb = weight_buft.find(name); wb != weight_buft.end()) {
                buft = wb->second;
            }
            t.signs = make_signs(width, buft);
            if (t.signs == nullptr) {
                LLAMA_LOG_WARN("%s: prism.hadamard has no sign vector of width %lld for %s\n",
                        __func__, (long long) width, name.c_str());
                return;
            }
        }
        if (gdn_v_grouped && name.find(".ssm_out.") != std::string::npos) {
            const int64_t n_v = (int64_t) model.hparams.ssm_dt_rank;
            const int64_t n_k = (int64_t) model.hparams.ssm_n_group;
            if (n_v > 0 && n_k > 0 && width % n_v == 0 && n_v % n_k == 0) {
                t.perm_hd  = width / n_v;
                t.perm_nk  = n_k;
                t.perm_rep = n_v / n_k;
            }
        }
    };

    for (const auto & name : weight_names) {
        fill_transform(name);
    }
    for (const auto & name : inverse_names) {
        fill_transform(name);
    }

    // The class-path graphs (llama_model_base, llm_graph_context) do not read hadamard_map: they read model.hdmd, upstream's
    // layout (names, sign data, rot/inv transforms keyed by the model's weight tensors). Record the same data there, from the
    // entries that really filled above, so both graph paths rotate with identical matrices and signs
    // (row S69-ENGREV-UPSTREAM-MERGE-PRISM-HADAMARD-TWO-IMPLEMENTATIONS-2026-11-04-01). Upstream: load_hparams_hadamard +
    // load_tensors_hadamard (ggml-org/llama.cpp src/llama-model.cpp, MIT).
    if (dynamic_cast<llama_model_base *>(&model) != nullptr) {
        const auto filled = [&](const std::string & name) {
            const auto it = model.hadamard_map.find(name);
            return it != model.hadamard_map.end() && it->second.block_size > 0 &&
                   (signs_by_width.empty() || it->second.signs != nullptr);
        };
        bool tied_output = false;
        ml.get_key("prism.hadamard.tied_output", tied_output, false);
        for (const auto & name : weight_names) {
            if (filled(name)) {
                model.hdmd.weight_blocks.emplace(name, block_size);
            }
        }
        for (const auto & name : inverse_names) {
            if (filled(name)) {
                model.hdmd.inverse_blocks.emplace(name, block_size);
            }
        }
        model.hdmd.tied_output   = tied_output;
        model.hdmd.gdn_v_grouped = gdn_v_grouped;
        for (const auto & [width, signs] : signs_by_width) {
            auto & vec = model.hdmd.sign_data[(uint32_t) width];
            vec.resize(signs.size());
            for (size_t j = 0; j < signs.size(); ++j) {
                vec[j] = (int32_t) signs[j];
            }
        }
        llama_hdmd_fill(model);
    }

    LLAMA_LOG_INFO("%s: PrismML Hadamard enabled: %zu rotated weights, %u sign widths, %zu inverse rows, block %u\n",
            __func__, weight_names.size(), (uint32_t) signs_by_width.size(), inverse_names.size(), block_size);
}
