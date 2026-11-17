#include "arg.h"
#include <sstream>    // --token-file parsing
#include <stdexcept>

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "ggml-vram-spill.h"
#include "download.h"
#include "../src/offload_attention_dense.h" // --offload-attn-dense-layers list/last-N grammar (syntax check at parse time)
#include "json-schema-to-grammar.h"
#include "json.h"
#include "llama.h"
#include "log.h"
#include "lora-merge.h"
#include "sampling.h"
#include "speculative.h"
#include "preset.h"
#include "watermark.h"
#include "../tools/server/server-ollama-store.h"
#include "../tools/server/server-ollama-gotmpl.h"
#include "../tools/server/server-ratelimit.h"
#include "gguf.h"
#include "imatrix-hessian.h" // --hessian-block bounds

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

#include <algorithm>
#include <charconv> // std::from_chars (--moe-cache-mib)
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <list>
#include <numeric>
#include <regex>
#include <set>
#include <string>
#include <system_error>
#include <thread> // for hardware_concurrency
#include <vector>

#ifndef __EMSCRIPTEN__
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#   if !defined(PATH_MAX)
#   define PATH_MAX MAX_PATH
#   endif
#elif defined(_AIX)
#include <sys/limits.h>
#else
#include <sys/syslimits.h>
#endif
#endif

#define LLAMA_MAX_URL_LENGTH 2084 // Maximum URL Length in Chrome: 2083

using json = common_json;
using namespace common_arg_utils;

// defined further down; the mmproj default-device pass (common_params_handle_model / postprocess) uses it first
static std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value);

static std::initializer_list<enum llama_example> mmproj_examples = {
    LLAMA_EXAMPLE_MTMD,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CLI,
    LLAMA_EXAMPLE_TTS,
    LLAMA_EXAMPLE_IMATRIX, // --image-calib
};

static std::string read_file(const std::string & fname) {
    std::ifstream file(fname);
    if (!file) {
        throw std::runtime_error(string_format("error: failed to open file '%s'\n", fname.c_str()));
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

static const std::vector<common_arg> & get_common_arg_defs() {
    static const std::vector<common_arg> options = [] {
        common_params params;
        auto ctx = common_params_parser_init(params, LLAMA_EXAMPLE_SERVER, nullptr);
        return ctx.options;
    }();
    return options;
}

common_arg & common_arg::set_examples(std::initializer_list<enum llama_example> examples) {
    this->examples = examples;
    return *this;
}

common_arg & common_arg::set_excludes(std::initializer_list<enum llama_example> excludes) {
    this->excludes = excludes;
    return *this;
}

common_arg & common_arg::set_env(const char * env) {
    help = help + "\n(env: " + env + ")";
    this->env = env;
    return *this;
}

common_arg & common_arg::set_sampling() {
    is_sampling = true;
    return *this;
}

common_arg & common_arg::set_spec() {
    is_spec = true;
    return *this;
}

common_arg & common_arg::set_preset_only() {
    is_preset_only = true;
    return *this;
}

bool common_arg::in_example(enum llama_example ex) {
    return examples.find(ex) != examples.end();
}

bool common_arg::is_exclude(enum llama_example ex) {
    return excludes.find(ex) != excludes.end();
}

bool common_arg::get_value_from_env(std::string & output) const {
    if (env == nullptr) return false;
    if (!args_neg.empty()) {
        // for compatibility, we need to check LLAMA_ARG_NO_ env as well
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        char * neg_value = std::getenv(neg_env.c_str());
        if (neg_value) {
            output = "0"; // falsey
            return true;
        }
    }
    char * value = std::getenv(env);
    if (value) {
        output = value;
        return true;
    }
    return false;
}

bool common_arg::has_value_from_env() const {
    if (env != nullptr && !args_neg.empty()) {
        // for compatibility, we need to check LLAMA_ARG_NO_ env as well
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        if (std::getenv(neg_env.c_str())) {
            return true;
        }
    }
    return env != nullptr && std::getenv(env);
}

static std::vector<std::string> break_str_into_lines(std::string input, size_t max_char_per_line) {
    std::vector<std::string> result;
    std::istringstream iss(input);
    std::string line;
    auto add_line = [&](const std::string& l) {
        if (l.length() <= max_char_per_line) {
            result.push_back(l);
        } else {
            std::istringstream line_stream(l);
            std::string word, current_line;
            while (line_stream >> word) {
                if (current_line.length() + !current_line.empty() + word.length() > max_char_per_line) {
                    if (!current_line.empty()) result.push_back(current_line);
                    current_line = word;
                } else {
                    current_line += (!current_line.empty() ? " " : "") + word;
                }
            }
            if (!current_line.empty()) result.push_back(current_line);
        }
    };
    while (std::getline(iss, line)) {
        add_line(line);
    }
    return result;
}

std::string common_arg::to_string() const {
    // params for printing to console
    const static int n_leading_spaces = 40;
    const static int n_char_per_line_help = 70; // TODO: detect this based on current console
    std::string leading_spaces(n_leading_spaces, ' ');

    std::ostringstream ss;
    auto all_args = get_args(); // also contains args_neg
    for (const auto & arg : all_args) {
        if (arg == all_args.front()) {
            if (all_args.size() == 1) {
                ss << arg;
            } else {
                // first arg is usually abbreviation, we need padding to make it more beautiful
                auto tmp = std::string(arg) + ", ";
                auto spaces = std::string(std::max(0, 7 - (int)tmp.size()), ' ');
                ss << tmp << spaces;
            }
        } else {
            ss << arg << (arg != all_args.back() ? ", " : "");
        }
    }
    if (value_hint) ss << " " << value_hint;
    if (value_hint_2) ss << " " << value_hint_2;
    if (ss.tellp() > n_leading_spaces - 3) {
        // current line is too long, add new line
        ss << "\n" << leading_spaces;
    } else {
        // padding between arg and help, same line
        ss << std::string(leading_spaces.size() - ss.tellp(), ' ');
    }
    const auto help_lines = break_str_into_lines(help, n_char_per_line_help);
    for (const auto & line : help_lines) {
        ss << (&line == &help_lines.front() ? "" : leading_spaces) << line << "\n";
    }
    return ss.str();
}

std::vector<std::string> common_arg::get_args() const {
    std::vector<std::string> result;
    for (const auto & arg : args) {
        result.push_back(std::string(arg));
    }
    for (const auto & arg : args_neg) {
        result.push_back(std::string(arg));
    }
    return result;
}

std::vector<std::string> common_arg::get_env() const {
    std::vector<std::string> result;
    if (env) {
        result.push_back(std::string(env));
    }
    if (!args_neg.empty() && env) {
        // for compatibility, we need to add LLAMA_ARG_NO_ variant
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        result.push_back(neg_env);
    }
    return result;
}

//
// utils
//

// Helper function to parse tensor buffer override strings
static void parse_tensor_buffer_overrides(const std::string & value, std::vector<llama_model_tensor_buft_override> & overrides) {
    ggml_backend_load_all();

    std::map<std::string, ggml_backend_buffer_type_t> buft_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        if (buft) {
            buft_list[ggml_backend_buft_name(buft)] = buft;
        }
        // "CUDA<n>_Spill" (see --vram-spill): published by the CUDA backend through its registry; absent on other builds.
        if (auto * reg = ggml_backend_dev_backend_reg(dev)) {
            using spill_fn = ggml_backend_buffer_type_t (*)(int);
            if (auto fn = (spill_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_spill_buffer_type")) {
                for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); ++j) {
                    if (ggml_backend_reg_dev_get(reg, j) == dev) {
                        if (auto * sbuft = fn((int) j)) {
                            buft_list[ggml_backend_buft_name(sbuft)] = sbuft;
                        }
                    }
                }
            }
        }
    }

    for (const auto & override : string_split<std::string>(value, ',')) {
        std::string::size_type pos = override.find('=');
        if (pos == std::string::npos) {
            throw std::invalid_argument("invalid value");
        }
        std::string tensor_name = override.substr(0, pos);
        std::string buffer_type = override.substr(pos + 1);

        if (buft_list.find(buffer_type) == buft_list.end()) {
            printf("Available buffer types:\n");
            for (const auto & it : buft_list) {
                printf("  %s\n", ggml_backend_buft_name(it.second));
            }
            throw std::invalid_argument("unknown buffer type");
        }
        // keep strings alive and avoid leaking memory by storing them in a static vector
        static std::list<std::string> buft_overrides;
        buft_overrides.push_back(tensor_name);
        overrides.push_back({buft_overrides.back().c_str(), buft_list.at(buffer_type)});
    }
}

static std::string clean_file_name(const std::string & fname) {
    std::string clean_fname = fname;
    string_replace_all(clean_fname, "\\", "_");
    string_replace_all(clean_fname, "/", "_");
    return clean_fname;
}

struct handle_model_result {
    bool found_mmproj = false;
    common_params_model mmproj;

    bool found_mtp = false;
    common_params_model mtp;

    bool found_preset = false;
    std::string preset_path;
};

const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_F8_E4M3, // mainline 202c1a229 spelling: -ctk f8_e4m3 -ctv f8_e4m3
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    // RESTORED 2026-09-27: the fork's pre-merge kv_cache_type_from_str (common.cpp @ 35f0cac75:4255) accepted
    // these two fork types; the merged upstream table dropped them, so `-ctk q8_KV` / `-ctk q6_0` answered
    // "Invalid cache type" while ggml (ggml.c:987, :1695) and the iqk + CUDA flash-attention kernels support both.
    GGML_TYPE_Q6_0,
    GGML_TYPE_Q8_KV,
    // TurboQuant KV (ggml.h:445-447). The fused flash-attention paths (CUDA fattn-turbo.cuh; Vulkan, Metal and SYCL through
    // the Q rotation / output un-rotation around their ordinary kernels) rotate Q once and un-rotate V once, so the graph needs
    // nothing else -- but NO parser, before or after the merge (common.cpp @ 35f0cac75:4255 had none either), accepted these
    // names, so the kernels were unreachable from any CLI and no model-level number could exist. turbo4 (the 4-bit PolarQuant
    // layout, 66-byte block) is listed since 2026-10-05: the CPU writer was repaired (dd765fbf0, it had been excluded for task
    // #278, a CPU inverse that did not invert its forward rotation) and the CUDA writer and fused read path carry it.
    GGML_TYPE_TURBO3_0,
    GGML_TYPE_TURBO2_0,
    GGML_TYPE_TURBO4_0,
};

static ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types) {
        if (ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

static std::string get_all_kv_cache_types() {
    std::ostringstream msg;
    bool first = true;
    for (const auto & type : kv_cache_types) {
        // a type with per-row metadata (q8_KV) is still PARSED, so llama_init_from_model answers it with its own explanation
        // (src/llama-context.cpp), but it is not advertised: the context refuses it (Thireus f228595b)
        if (ggml_internal_get_type_traits(type).row_meta_size > 0) {
            continue;
        }
        msg << (first ? "" : ", ") << ggml_type_name(type);
        first = false;
    }
    return msg.str();
}

static bool parse_bool_value(const std::string & value) {
    if (is_truthy(value)) {
        return true;
    } else if (is_falsey(value)) {
        return false;
    } else {
        throw std::invalid_argument("invalid boolean value");
    }
}

[[noreturn]] static void arg_removed(const std::string & msg) {
    throw std::invalid_argument("the argument has been removed. " + msg);
}

//
// common_models_handler
//

static std::string get_default_local_path(const std::string & url) {
    auto f = string_split<std::string>(url, '#').front();
    f = string_split<std::string>(f, '?').front();
    return fs_get_cache_file(string_split<std::string>(f, '/').back()).string();
}

// ADAPTED (S69-ARGCPP-SPEC-SHAPE-MISMATCH-2026-09-08-01): theirs compared params.speculative.types
// (a std::vector<common_speculative_type>) against a 1-element {NONE} sentinel. The staged header
// has no .types vector -- it carries a resolved multi-stage chain instead (.type + .stages,
// common_params_speculative::has_stage_chain()/get_resolved_stages() in common.h/common.cpp).
// has_stage_chain() returns false exactly when no stage was configured, the same condition
// "== {NONE}" was checking for.
static bool spec_types_is_default(const common_params & params) {
    return !params.speculative.has_stage_chain();
}

// ADAPTED, same row: theirs assigned/pushed onto params.speculative.types (missing vector)
// directly by type value; header's real chain container is .stages
// (vector<common_speculative_stage_params>), each element type-only unless a stage override
// applies. These two helpers give arg.cpp's call sites the same "set to this one type" /
// "append this type" shape without duplicating the wrapping logic at every call site.
static void spec_stages_set(common_params & params, common_speculative_type type) {
    params.speculative.stages = { common_speculative_stage_params{ /* .type = */ type } };
}
static void spec_stages_push(common_params & params, common_speculative_type type) {
    params.speculative.stages.push_back(common_speculative_stage_params{ /* .type = */ type });
}
static void spec_stages_push_all(common_params & params, const std::vector<common_speculative_type> & types) {
    for (const auto & t : types) {
        spec_stages_push(params, t);
    }
}

// --spec-self-draft-type TYPE (S69-IPEXLLM-PICK-SELF-SPECULATIVE-DRAFT-2026-09-29-01). IPEX-LLM's self-speculative decoding (Apache-2.0,
// ipex_llm/transformers/model.py:366-375, read by the row's reviewers) loads the SAME checkpoint a second time as a 4-bit model and uses it as
// the draft of the full-precision target. Here the draft is a lower-bit requantisation of the target GGUF, produced once by llama_model_quantize
// and cached next to the target (<stem>.selfdraft-<type>.gguf, reused while it is newer than the target), then handed to the ordinary draft path.
// An explicit draft model always wins; with no --spec-type given the stage chain becomes the plain draft stage.
struct self_draft_type_entry {
    const char *    name;
    llama_ftype     ftype;
};
static const self_draft_type_entry SELF_DRAFT_TYPES[] = {
    { "q2_k",   LLAMA_FTYPE_MOSTLY_Q2_K   }, { "q3_k_m", LLAMA_FTYPE_MOSTLY_Q3_K_M }, { "q4_0",   LLAMA_FTYPE_MOSTLY_Q4_0   },
    { "q4_k_s", LLAMA_FTYPE_MOSTLY_Q4_K_S }, { "q4_k_m", LLAMA_FTYPE_MOSTLY_Q4_K_M }, { "q5_0",   LLAMA_FTYPE_MOSTLY_Q5_0   },
    { "q5_k_m", LLAMA_FTYPE_MOSTLY_Q5_K_M }, { "q6_k",   LLAMA_FTYPE_MOSTLY_Q6_K   }, { "q8_0",   LLAMA_FTYPE_MOSTLY_Q8_0   },
    { "iq4_nl", LLAMA_FTYPE_MOSTLY_IQ4_NL }, { "iq4_xs", LLAMA_FTYPE_MOSTLY_IQ4_XS },
};

static void common_derive_self_draft(common_params & params) {
    const std::string type = params.speculative.self_draft_type;
    if (type.empty()) {
        return;
    }
    if (params.speculative.has_dft()) {
        LOG_WRN("%s: --spec-self-draft-type %s ignored: a draft model is already set (%s)\n", __func__, type.c_str(), params.speculative.dft_path().c_str());
        return;
    }
    const std::string & src = params.model.path;
    std::error_code ec;
    if (src.empty() || !std::filesystem::exists(src, ec)) {
        LOG_WRN("%s: --spec-self-draft-type %s needs a local target model file; got '%s', no draft derived\n", __func__, type.c_str(), src.c_str());
        return;
    }
    if (src.find("-of-") != std::string::npos) {
        throw std::invalid_argument("error: --spec-self-draft-type does not support a split (sharded) target model: quantize it to one file and pass -md\n");
    }
    const self_draft_type_entry * entry = nullptr;
    std::string known;
    for (const auto & e : SELF_DRAFT_TYPES) {
        known += std::string(known.empty() ? "" : ", ") + e.name;
        if (type == e.name) {
            entry = &e;
        }
    }
    if (entry == nullptr) {
        throw std::invalid_argument("error: --spec-self-draft-type '" + type + "' is not one of: " + known + "\n");
    }
    const std::filesystem::path sp(src);
    const std::filesystem::path out = sp.parent_path() / (sp.stem().string() + ".selfdraft-" + type + ".gguf");
    const auto src_time = std::filesystem::last_write_time(sp, ec);
    const bool fresh = std::filesystem::exists(out, ec) && std::filesystem::file_size(out, ec) > 0 &&
                       std::filesystem::last_write_time(out, ec) >= src_time;
    if (fresh) {
        LOG_INF("%s: reusing the self-draft %s (newer than the target)\n", __func__, out.string().c_str());
    } else {
        LOG_INF("%s: deriving a %s self-draft from %s -> %s (once; cached next to the target)\n", __func__, type.c_str(), src.c_str(), out.string().c_str());
        llama_backend_init();
        llama_model_quantize_params qp = llama_model_quantize_default_params();
        qp.ftype            = entry->ftype;
        qp.nthread          = params.cpuparams.n_threads > 0 ? params.cpuparams.n_threads : 0;
        qp.allow_requantize = true;   // the target may itself be quantized (Q8_0 is the typical case)
        const std::string tmp = out.string() + ".tmp";
        const uint32_t rc = llama_model_quantize(src.c_str(), tmp.c_str(), &qp);
        if (rc != 0) {
            throw std::runtime_error("error: deriving the self-draft failed (llama_model_quantize returned " + std::to_string(rc) + ")\n");
        }
        std::filesystem::rename(tmp, out, ec);
        if (ec) {
            throw std::runtime_error("error: could not move the derived self-draft into place: " + ec.message() + "\n");
        }
    }
    params.speculative.mparams_dft.path = out.string();
    if (spec_types_is_default(params)) {
        spec_stages_set(params, COMMON_SPECULATIVE_TYPE_DRAFT);
    }
}

// --spec-type SPEC[:key=value][,key=value...][,SPEC2[:...]] : the fork's error messages for the REMOVED --suffix-corpus / --suffix-max-depth /
// --suffix-pattern-len flags tell the user to write "--spec-type suffix:suffix_min_match_len=<n>", but the handler below only ever accepted bare type
// names, so those three stage parameters had no way in from the command line at all (found 2026-10-05 while writing the acceptance guard). A token with a
// ':' starts a stage and carries its first key=value; a later token with '=' and no ':' adds one more key=value to the stage before it; any other token
// is a bare type name, exactly as before.
static void spec_stage_apply_kv(common_speculative_stage_params & sp, const std::string & kv) {
    const size_t eq = kv.find('=');
    if (eq == std::string::npos || eq == 0) {
        throw std::invalid_argument("--spec-type: expected key=value, got '" + kv + "'");
    }
    const std::string k = kv.substr(0, eq);
    const std::string v = kv.substr(eq + 1);
    // whole-string numeric parse with a range, so "abc", "7x" and a value that would wrap a 16-bit field are refused by name
    // instead of surfacing as a bare "stoi" or a silently truncated number
    auto num = [&](long long lo, long long hi) -> long long {
        size_t used = 0;
        long long x = 0;
        try {
            x = std::stoll(v, &used);
        } catch (const std::exception &) {
            used = 0;
        }
        if (used == 0 || used != v.size() || x < lo || x > hi) {
            throw std::invalid_argument("--spec-type: " + k + " needs an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "], got '" + v + "'");
        }
        return x;
    };
    auto flt = [&](float lo, float hi) -> float {
        size_t used = 0;
        float x = 0.0f;
        try {
            x = std::stof(v, &used);
        } catch (const std::exception &) {
            used = 0;
        }
        if (used == 0 || used != v.size() || !(x >= lo && x <= hi)) {
            throw std::invalid_argument("--spec-type: " + k + " needs a number in [" + std::to_string(lo) + ", " + std::to_string(hi) + "], got '" + v + "'");
        }
        return x;
    };
    if      (k == "n_max")                sp.n_max                = (int32_t)  num(0, 1 << 20);
    else if (k == "n_min")                sp.n_min                = (int32_t)  num(0, 1 << 20);
    else if (k == "p_min")                sp.p_min                = flt(0.0f, 1.0f);
    else if (k == "mtp_heads")            sp.mtp_heads            = (int32_t)  num(0, 64);
    else if (k == "ngram_size_n")         sp.ngram_size_n         = (uint16_t) num(1, 65535);
    else if (k == "ngram_size_m")         sp.ngram_size_m         = (uint16_t) num(1, 65535);
    else if (k == "ngram_min_hits")       sp.ngram_min_hits       = (uint16_t) num(1, 65535);
    else if (k == "suffix_min_match_len") sp.suffix_min_match_len = (int32_t)  num(1, 1 << 20);
    else if (k == "suffix_max_depth")     sp.suffix_max_depth     = (int32_t)  num(0, 1 << 20);
    else if (k == "suffix_max_nodes")     sp.suffix_max_nodes     = (int64_t)  num(0, 1ll << 40);
    else if (k == "suffix_corpus")        sp.suffix_corpus        = v;
    else {
        throw std::invalid_argument("--spec-type: unknown stage parameter '" + k + "' (n_max, n_min, p_min, mtp_heads, ngram_size_n, ngram_size_m, ngram_min_hits, "
                                    "suffix_min_match_len, suffix_max_depth, suffix_max_nodes, suffix_corpus)");
    }
}

static void spec_type_arg_apply(common_params & params, const std::string & value) {
    std::vector<common_speculative_stage_params> out;
    for (const auto & tok : string_split<std::string>(value, ',')) {
        const size_t colon = tok.find(':');
        if (colon == std::string::npos && tok.find('=') != std::string::npos) {
            if (out.empty()) {
                throw std::invalid_argument("--spec-type: '" + tok + "' has no stage before it");
            }
            spec_stage_apply_kv(out.back(), tok);
            continue;
        }
        const std::string name = colon == std::string::npos ? tok : tok.substr(0, colon);
        const auto types = common_speculative_types_from_names({ name });
        common_speculative_stage_params sp;
        sp.type = types.front();
        out.push_back(sp);
        if (colon != std::string::npos && colon + 1 < tok.size()) {
            spec_stage_apply_kv(out.back(), tok.substr(colon + 1));
        }
    }
    for (const auto & sp : out) {
        params.speculative.stages.push_back(sp);
    }
}

common_models_handler common_models_handler_init(const common_params & params, llama_example curr_ex) {
    common_download_hf_plan plan;
    common_download_hf_plan plan_spec;
    common_download_opts opts;

    // ADAPTED (S69-ARGCPP-SPEC-SHAPE-MISMATCH-2026-09-08-01): theirs scanned params.speculative.types
    // (missing vector); has_stage_type() over the resolved stage chain is the header's real
    // equivalent (already used the same way by speculative.cpp's own adapted functions).
    const bool spec_type_draft_mtp    = params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);
    const bool spec_type_draft_dflash = params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_DFLASH);
    const bool spec_type_draft_eagle3 = params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_EAGLE3);
    const bool spec_type_draft_dspark = params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_DSPARK);

    // only download mmproj if the current example is using it
    bool use_mmproj = false;
    for (const auto & ex : mmproj_examples) {
        if (curr_ex == ex) {
            use_mmproj = true;
            break;
        }
    }
    if (curr_ex == LLAMA_EXAMPLE_DOWNLOAD) {
        use_mmproj = true;
    }

    opts.bearer_token    = params.hf_token;
    opts.offline         = params.offline;
    opts.chunk_concurrency = params.download_connections;
    opts.chunk_size        = (size_t) std::max<int32_t>(0, params.download_chunk_mib) << 20;
    opts.download_mtp   = spec_type_draft_mtp;
    opts.download_eagle3 = spec_type_draft_eagle3;
    opts.download_dflash = spec_type_draft_dflash;
    opts.download_dspark = spec_type_draft_dspark;
    opts.download_mmproj = use_mmproj && !params.no_mmproj
                        && params.mmproj.path.empty() && params.mmproj.url.empty();

    if (!params.model.hf_repo.empty()) {
        // ADAPTED (S69-ARGCPP-MODEL-NESTED-VS-FLAT-2026-09-08-01): common_download_get_hf_plan
        // (common/download.h, forbidden file, unmodified) takes a real common_params_model.
        // params.model is the fork's flat std::string path; build a throwaway
        // common_params_model from the flat sibling fields to satisfy the (unowned) signature --
        // read-only here, params is `const common_params &` in this function.
        common_params_model model_for_plan;
        model_for_plan.path        = params.model.path;
        model_for_plan.url         = params.model.url;
        model_for_plan.hf_repo     = params.model.hf_repo;
        model_for_plan.hf_file     = params.model.hf_file;
        model_for_plan.hf_revision = params.model.hf_revision;
        model_for_plan.docker_repo = params.model.docker_repo;
        plan = common_download_get_hf_plan(model_for_plan, opts);
    }

    if (!params.speculative.mparams_dft.hf_repo.empty()) {
        // without a requested type, discover every sidecar the draft repo ships to infer the type later
        auto opts_spec = opts;
        if (spec_types_is_default(params)) {
            opts_spec.download_mtp    = true;
            opts_spec.download_dflash = true;
            opts_spec.download_eagle3 = true;
            opts_spec.download_dspark = true;
        }
        plan_spec = common_download_get_hf_plan(params.speculative.mparams_dft, opts_spec);
    }

    return common_models_handler{plan, plan_spec, opts};
}

bool common_models_handler_is_preset_repo(const common_models_handler & handler) {
    return !handler.plan.preset.url.empty();
}

// ADAPTED (out-of-band from the speculative escalation, same underlying cause -- discovered
// making arg.cpp compile end to end -- S69-ARGCPP-MODEL-NESTED-VS-FLAT-2026-09-08-01): theirs
// took a `const common_params_model & model` (nested .url/.path). The staged common.h keeps
// common_params::model as a FLAT std::string path plus separate flat model_url/hf_repo/hf_file
// (a deliberate sibling decision, see common.h's own comment above the `model` field) -- there
// is no nested common_params_model object to bind a reference to for params.model itself
// (mmproj and speculative.mparams_dft ARE still real common_params_model objects and are
// untouched below). Retyped to the two flat strings this function actually reads/writes.
static std::vector<common_download_task> build_url_tasks(const std::string & url, const std::string & path, common_download_opts opts) {
    auto parts = common_download_get_all_parts(url);
    std::vector<common_download_task> tasks;

    // single-part: download straight to model.path if the user gave one (-m), else the cache default
    if (parts.size() == 1) {
        common_download_task task;
        task.url        = parts[0];
        task.local_path = path.empty() ? get_default_local_path(parts[0]) : path;
        task.opts       = opts;
        tasks.push_back(std::move(task));
        return tasks;
    }

    // multi-part: place each part under the user's -m directory (if given), else the cache default
    std::string base_dir;
    if (!path.empty()) {
        auto pos = path.rfind('/');
        base_dir = pos == std::string::npos ? std::string(".") : path.substr(0, pos);
    }

    for (const auto & part : parts) {
        common_download_task task;
        task.url  = part;
        task.opts = opts;

        std::string local = get_default_local_path(part);
        if (!base_dir.empty()) {
            auto pos = local.rfind('/');
            std::string name = pos == std::string::npos ? local : local.substr(pos + 1);
            local = base_dir + "/" + name;
        }
        task.local_path = local;
        tasks.push_back(std::move(task));
    }
    return tasks;
}

void common_models_handler_apply(common_models_handler & handler, common_params & params, common_download_callback * callback) {
    std::vector<common_download_task> tasks;

    auto & plan      = handler.plan;
    auto & plan_spec = handler.plan_spec;

    auto opts = handler.opts; // copy
    opts.callback = callback;

    // another process may hold the cache lock during finalization:
    // let the download callback cancel that wait too
    auto finalize = [callback](const hf_cache::hf_file & f) {
        return hf_cache::finalize_file(f, [callback]() {
            return !(callback && callback->is_cancelled());
        });
    };

    // handle plain "url" if needed
    // ADAPTED, same row as build_url_tasks above: params.model is flat (std::string model +
    // std::string model_url), not a common_params_model -- a second, flat-field-typed variant
    // handles it; mmproj/speculative.mparams_dft are still real common_params_model objects
    // and keep using the original struct-typed lambda unchanged.
    auto handle_url = [&](common_params_model & model) {
        if (!model.url.empty()) {
            if (model.path.empty()) {
                model.path = get_default_local_path(model.url);
            }
        }
    };
    auto handle_url_flat = [&](std::string & url, std::string & path) {
        if (!url.empty()) {
            if (path.empty()) {
                path = get_default_local_path(url);
            }
        }
    };
    handle_url_flat(params.model.url, params.model.path);
    handle_url(params.mmproj);
    handle_url(params.speculative.mparams_dft);

    // optionally, if docker repo is set, resolve it
    if (!params.model.docker_repo.empty()) {
        params.model.url = common_docker_resolve_model(params.model.docker_repo);
        params.model.path     = get_default_local_path(params.model.url);
    }

    // handle plain "url" tasks (non-hf)
    if (!params.model.url.empty()) {
        auto url_tasks = build_url_tasks(params.model.url, params.model.path, opts);
        // the first part is what gets loaded, so point params.model at it
        if (!url_tasks.empty()) {
            std::string first_path = url_tasks.front().local_path;
            url_tasks.front().on_done = [&, first_path]() { params.model.path = first_path; };
        }
        for (auto & task : url_tasks) {
            tasks.push_back(std::move(task));
        }
    }
    if (!params.mmproj.url.empty()) {
        common_download_task task;
        task.url        = params.mmproj.url;
        task.local_path = params.mmproj.path;
        task.opts       = opts;
        tasks.push_back(task);
    }
    bool had_spec_url = false;
    if (!params.speculative.mparams_dft.url.empty()) {
        common_download_task task;
        task.url        = params.speculative.mparams_dft.url;
        task.local_path = params.speculative.mparams_dft.path;
        task.opts       = opts;
        tasks.push_back(task);
        had_spec_url = true;
    }

    // handle hf_plan tasks
    // ADAPTED, same row as build_url_tasks/handle_url above: only ever writes the primary
    // file's resolved path into a single string field (was `model.path` on a
    // common_params_model&) -- retyped to that string directly so both real
    // common_params_model::path (mparams_dft, passed by member) and the flat params.model
    // string (passed directly) work at their call sites unchanged in shape.
    auto add_tasks = [&opts, &tasks, &finalize](const hf_cache::hf_files  & model_files,
                                    const hf_cache::hf_file    & primary,
                                    std::string                & path_out) {
        for (size_t i = 0; i < model_files.size(); ++i) {
            auto & model_file = model_files[i];
            bool is_primary = (model_file.path == primary.path);
            tasks.emplace_back(model_file, opts, [&, is_primary]() {
                if (is_primary) {
                    // the primary file is the first split (00001-of), use it as model path
                    path_out = finalize(model_file);
                } else {
                    finalize(model_file);
                }
            });
        }
    };

    // an explicit draft file selection (e.g. -md with -hfd) disables the sidecar resolution of the draft repo
    if (!params.speculative.mparams_dft.hf_file.empty()) {
        plan_spec.mtp    = {};
        plan_spec.dflash = {};
        plan_spec.eagle3 = {};
        plan_spec.dspark = {};
    }

    // infer the speculative type from the sidecar shipped by the draft repo when none is requested
    if (spec_types_is_default(params)) {
        if (!plan_spec.mtp.local_path.empty()) {
            spec_stages_set(params, COMMON_SPECULATIVE_TYPE_MTP);
            plan_spec.dspark = {};
            plan_spec.dflash = {};
            plan_spec.eagle3 = {};
        } else if (!plan_spec.dspark.local_path.empty()) {
            // dspark outranks dflash, its sidecar carries the extra Markov head
            spec_stages_set(params, COMMON_SPECULATIVE_TYPE_DSPARK);
            plan_spec.dflash = {};
            plan_spec.eagle3 = {};
        } else if (!plan_spec.dflash.local_path.empty()) {
            spec_stages_set(params, COMMON_SPECULATIVE_TYPE_DFLASH);
            plan_spec.eagle3 = {};
        } else if (!plan_spec.eagle3.local_path.empty()) {
            spec_stages_set(params, COMMON_SPECULATIVE_TYPE_EAGLE3);
        }
    }

    // infer the speculative type from the draft GGUF metadata when none is requested
    // note: reads only the first split - sharded drafts need an explicit --spec-type
    if (spec_types_is_default(params) && !params.speculative.mparams_dft.path.empty()) {
        const auto types_gguf = common_speculative_types_from_gguf(params.speculative.mparams_dft.path);
        if (!types_gguf.empty()) {
            spec_stages_push_all(params, types_gguf);
        }
    }

    // a plain causal draft GGUF carries no type metadata (only MTP / DFlash / DSpark are detected above), so
    // `-md X` alone left the chain empty: the draft model loaded (+1 GB VRAM) and nothing was ever drafted
    // (MEASURED 2026-10-27 on :8081, row S69-FORK-SPEC-MD-WITHOUT-TYPE-LOADS-DRAFT-NEVER-DRAFTS-2026-10-27-01).
    // A named draft model with no type requested means a draft-model stage, as the self-draft path above sets.
    // "No type requested" is an EMPTY stage list, not spec_types_is_default(): an explicit `--spec-type none`
    // stores a NONE stage, which has_stage_chain() filters out, so spec_types_is_default() reads it as default
    // and would override the opt-out (MEASURED 2026-10-27, test-spec-md-implicit-draft case 5).
    if (params.speculative.stages.empty() && !params.speculative.mparams_dft.path.empty()) {
        LOG_INF("%s: -md given without --spec-type and the GGUF names no type -> using a 'draft' stage\n", __func__);
        spec_stages_set(params, COMMON_SPECULATIVE_TYPE_DRAFT);
    }

    // when a sidecar type is requested, the draft repo resolves to its sidecar instead of a full model
    const bool spec_sidecar_found = !plan_spec.mtp.local_path.empty() ||
                                    !plan_spec.dflash.local_path.empty() ||
                                    !plan_spec.eagle3.local_path.empty() ||
                                    !plan_spec.dspark.local_path.empty();
    if (!plan_spec.mtp.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan_spec.mtp, opts, [&]() {
            // only use the discovered MTP head when no draft path is set yet
            if (params.speculative.mparams_dft.path.empty()) {
                params.speculative.mparams_dft.path = finalize(plan_spec.mtp);
            } else {
                finalize(plan_spec.mtp);
            }
        });
    }
    if (!plan_spec.dflash.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan_spec.dflash, opts, [&]() {
            // only use the discovered DFlash sidecar when no draft path is set yet
            if (params.speculative.mparams_dft.path.empty()) {
                params.speculative.mparams_dft.path = finalize(plan_spec.dflash);
            } else {
                finalize(plan_spec.dflash);
            }
        });
    }
    if (!plan_spec.eagle3.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan_spec.eagle3, opts, [&]() {
            // only use the discovered Eagle3 sidecar when no draft path is set yet
            if (params.speculative.mparams_dft.path.empty()) {
                params.speculative.mparams_dft.path = finalize(plan_spec.eagle3);
            } else {
                finalize(plan_spec.eagle3);
            }
        });
    }
    if (!plan_spec.dspark.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan_spec.dspark, opts, [&]() {
            // only use the discovered DSpark sidecar when no draft path is set yet
            if (params.speculative.mparams_dft.path.empty()) {
                params.speculative.mparams_dft.path = finalize(plan_spec.dspark);
            } else {
                finalize(plan_spec.dspark);
            }
        });
    }

    // a wired draft sidecar counts as an explicit draft for the main plan fallback below
    if (spec_sidecar_found) {
        had_spec_url = true;
    }

    // handle plan_spec (e.g. --spec-draft-hf)
    bool spec_plain_draft_download = false;
    if (!plan_spec.model_files.empty() && !had_spec_url && !spec_sidecar_found) {
        add_tasks(plan_spec.model_files, plan_spec.primary, params.speculative.mparams_dft.path);
        had_spec_url = true;
        spec_plain_draft_download = true;
    }

    if (!plan.model_files.empty()) {
        add_tasks(plan.model_files, plan.primary, params.model.path);
    }
    if (!plan.mmproj.local_path.empty()) {
        tasks.emplace_back(plan.mmproj, opts, [&]() {
            params.mmproj.path = finalize(plan.mmproj);
        });
    }
    if (!plan.mtp.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan.mtp, opts, [&]() {
            // only fall back to the discovered MTP head when no draft was explicitly provided
            if (params.speculative.mparams_dft.empty()) {
                params.speculative.mparams_dft.path = finalize(plan.mtp);
            } else {
                finalize(plan.mtp);
            }
        });
    }
    if (!plan.dflash.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan.dflash, opts, [&]() {
            // only fall back to the discovered DFlash sidecar when no draft was explicitly provided
            if (params.speculative.mparams_dft.empty()) {
                params.speculative.mparams_dft.path = finalize(plan.dflash);
            } else {
                finalize(plan.dflash);
            }
        });
    }
    if (!plan.eagle3.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan.eagle3, opts, [&]() {
            // only fall back to the discovered Eagle3 sidecar when no draft was explicitly provided
            if (params.speculative.mparams_dft.empty()) {
                params.speculative.mparams_dft.path = finalize(plan.eagle3);
            } else {
                finalize(plan.eagle3);
            }
        });
    }
    if (!plan.dspark.local_path.empty() && !had_spec_url) {
        tasks.emplace_back(plan.dspark, opts, [&]() {
            // only fall back to the discovered DSpark sidecar when no draft was explicitly provided
            if (params.speculative.mparams_dft.empty()) {
                params.speculative.mparams_dft.path = finalize(plan.dspark);
            } else {
                finalize(plan.dspark);
            }
        });
    }
    if (!plan.preset.local_path.empty()) {
        tasks.emplace_back(plan.preset, opts, [&]() {
            // if HF repo is a preset repo, we simply run server in router mode with the preset.ini file
            params.models_preset_hf = params.model.hf_repo; // only for showing a warning
            params.models_preset    = finalize(plan.preset);
            // clear the model so the server starts in router mode
            params.model.path.clear();
            params.model.hf_repo.clear();
            params.model.docker_repo.clear();
        });
    }

    // run all tasks in parallel
    if (!params.offline) {
        // if duplicated files are found, only download once (but still call on_done for each task)
        std::unordered_map<std::string, common_download_task *> unique_tasks;
        for (auto & task : tasks) {
            auto it = unique_tasks.find(task.local_path);
            if (it == unique_tasks.end()) {
                unique_tasks[task.local_path] = &task;
            }
        }
        std::vector<common_download_task> unique_tasks_vec;
        for (auto & pair : unique_tasks) {
            LOG_DBG("download task: %s -> %s\n", pair.second->url.c_str(), pair.second->local_path.c_str());
            unique_tasks_vec.push_back(*pair.second);
        }
        common_download_run_tasks(unique_tasks_vec);
    }

    // download successful, update params with the downloaded paths
    for (const auto & task : tasks) {
        if (task.on_done) {
            task.on_done();
        }
    }

    // same fallback for a downloaded draft model (--spec-draft-hf): its path is only known now. Only the plain
    // draft download counts; a target repo's MTP / DFlash / Eagle3 / DSpark head also lands in mparams_dft.path
    // here and must not be turned into a draft-model stage.
    if (spec_plain_draft_download && params.speculative.stages.empty() && !params.speculative.mparams_dft.path.empty()) {
        const auto types_gguf = common_speculative_types_from_gguf(params.speculative.mparams_dft.path);
        if (!types_gguf.empty()) {
            spec_stages_push_all(params, types_gguf);
        } else {
            LOG_INF("%s: downloaded draft model without --spec-type and the GGUF names no type -> using a 'draft' stage\n", __func__);
            spec_stages_set(params, COMMON_SPECULATIVE_TYPE_DRAFT);
        }
    }

    // the target path is final here: derive the self-draft (no-op unless --spec-self-draft-type was given)
    common_derive_self_draft(params);
}

//
// CLI argument parsing functions
//

// apply config files (if present), a later file overrides an earlier one:
// 1. system-wide: /etc/llama.cpp/config.ini (%PROGRAMDATA%\llama.cpp\config.ini on windows)
// 2. user-level: ${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini (%APPDATA%\llama.cpp\config.ini on windows)
static void common_params_apply_system_config(common_params & params, llama_example ex) {
    std::vector<std::filesystem::path> paths;

#if defined(_WIN32)
    const std::filesystem::path program_data = common_get_env("PROGRAMDATA");
    if (!program_data.empty()) {
        paths.push_back(program_data / "llama.cpp" / "config.ini");
    }
#else
    paths.push_back("/etc/llama.cpp/config.ini");
#endif

    try {
        paths.push_back(fs_get_config_directory() / "config.ini");
    } catch (const std::exception & e) {
        LOG_DBG("cannot read user-level config file, skipping: %s\n", e.what());
    }

    std::vector<std::filesystem::path> found;
    for (const auto & path : paths) {
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            found.push_back(path);
        }
    }
    if (found.empty()) {
        return;
    }

    common_preset_context ctx(ex);
    ctx.ignore_unknown_keys = true; // the same config file is shared by all programs
    for (const auto & path : found) {
        LOG_INF("using config file: %s\n", path.string().c_str());
        common_preset global;
        common_presets presets = ctx.load_from_ini(path, global);
        global.apply_to_params(params);
        auto it = presets.find(COMMON_PRESET_DEFAULT_NAME);
        if (it != presets.end()) {
            it->second.apply_to_params(params);
        }
    }
}

std::string common_hf_token_from_files() {
    auto env = [](const char * k) -> std::string {
        const char * v = std::getenv(k);
        return v ? std::string(v) : std::string();
    };
    std::vector<std::filesystem::path> cands;
    if (const std::string p = env("HF_TOKEN_PATH"); !p.empty()) {
        cands.emplace_back(p);
    }
    if (const std::string h = env("HF_HOME"); !h.empty()) {
        cands.emplace_back(std::filesystem::path(h) / "token");
    }
#ifdef _WIN32
    const std::string home = env("USERPROFILE");
#else
    const std::string home = env("HOME");
#endif
    if (!home.empty()) {
        cands.emplace_back(std::filesystem::path(home) / ".cache" / "huggingface" / "token");
    }
    for (const auto & c : cands) {
        std::ifstream f(c, std::ios::binary);
        if (!f) {
            continue;
        }
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const char * ws = " \t\r\n";
        const size_t b = s.find_first_not_of(ws);
        if (b == std::string::npos) {
            continue;
        }
        const size_t e = s.find_last_not_of(ws);
        return s.substr(b, e - b + 1);
    }
    return "";
}

static bool common_params_parse_ex(int argc, char ** argv, common_params_context & ctx_arg) {
    common_params & params = ctx_arg.params;

    // setup log directly from params.verbosity: see tools/cli/cli.cpp
    common_log_set_verbosity_thold(params.verbosity);

    // config file applies first, so env variables and CLI arguments override it
    common_params_apply_system_config(params, ctx_arg.ex);

    std::unordered_map<std::string, std::pair<common_arg *, bool>> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = {&opt, /* is_positive */ true};
        }
        for (const auto & arg : opt.args_neg) {
            arg_to_options[arg] = {&opt, /* is_positive */ false};
        }
    }

    // handle environment variables
    for (auto & opt : ctx_arg.options) {
        std::string value;
        if (opt.get_value_from_env(value)) {
            // an env variable is an explicit user setting too -- see args_seen in parse_cli_args
            for (const auto & a : opt.args) { params.args_seen.insert(a); }
            try {
                if (opt.handler_void && is_truthy(value)) {
                    opt.handler_void(params);
                }
                if (opt.handler_int) {
                    opt.handler_int(params, std::stoi(value));
                }
                if (opt.handler_bool) {
                    opt.handler_bool(params, parse_bool_value(value));
                }
                if (opt.handler_string) {
                    opt.handler_string(params, value);
                    continue;
                }
            } catch (std::exception & e) {
                throw std::invalid_argument(string_format(
                    "error while handling environment variable \"%s\": %s\n\n", opt.env, e.what()));
            }
        }
    }

    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };

    auto parse_cli_args = [&]() {
        std::set<std::string> seen_args;

        for (int i = 1; i < argc; i++) {
            const std::string arg_prefix = "--";

            std::string arg = argv[i];
            if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
                std::replace(arg.begin(), arg.end(), '_', '-');
            }
            if (arg_to_options.find(arg) == arg_to_options.end()) {
                throw std::invalid_argument(string_format("error: invalid argument: %s", arg.c_str()));
            }
            if (!seen_args.insert(arg).second) {
                const bool skip = (arg == "--spec-type");

                if (!skip) {
                    LOG_WRN("DEPRECATED: argument '%s' specified multiple times, use comma-separated values instead (only last value will be used)\n", arg.c_str());
                }
            }
            auto & tmp = arg_to_options[arg];
            auto opt = *tmp.first;
            bool is_positive = tmp.second;
            // args_seen is what common_params_sampling_apply_model_kv() consults so an explicit flag
            // beats the model's general.sampling.* recommendation. Nothing populated it, so the model
            // KV silently won: `--temp 0` ran at the GGUF's temp 1.0 with a random seed. Record every
            // spelling of the option, so `--temperature` protects `--temp` too.
            for (const auto & a : opt.args)     { params.args_seen.insert(a); }
            for (const auto & a : opt.args_neg) { params.args_seen.insert(a); }
            if (opt.has_value_from_env()) {
                fprintf(stderr, "warn: %s environment variable is set, but will be overwritten by command line argument %s\n", opt.env, arg.c_str());
            }
            try {
                if (opt.handler_void) {
                    opt.handler_void(params);
                    continue;
                }
                if (opt.handler_bool) {
                    opt.handler_bool(params, is_positive);
                    continue;
                }

                // arg with single value
                check_arg(i);
                std::string val = argv[++i];
                if (opt.handler_int) {
                    opt.handler_int(params, std::stoi(val));
                    continue;
                }
                if (opt.handler_string) {
                    opt.handler_string(params, val);
                    continue;
                }

                // arg with 2 values
                check_arg(i);
                std::string val2 = argv[++i];
                if (opt.handler_str_str) {
                    opt.handler_str_str(params, val, val2);
                    continue;
                }
            } catch (std::exception & e) {
                throw std::invalid_argument(string_format(
                    "error while handling argument \"%s\": %s\n\n"
                    "usage:\n%s\n\nto show complete usage, run with -h",
                    arg.c_str(), e.what(), opt.to_string().c_str()));
            }
        }

        // the deprecated load flags and --load-mode write the same field: say so when both are given (HEAD common/arg.cpp:1166-1175)
        auto has_arg = [&](std::initializer_list<const char *> names) {
            return std::any_of(names.begin(), names.end(), [&](const char * name) {
                return seen_args.count(name) > 0;
            });
        };
        if (has_arg({"-lm", "--load-mode"}) &&
            has_arg({"--mlock", "--mmap", "--no-mmap", "-dio", "--direct-io", "-ndio", "--no-direct-io"})) {
            LOG_WRN("DEPRECATED: `--load-mode` and `--mlock`/`--mmap`/`--direct-io` should not be combined; only the last flag on the command line will take effect\n");
        }
    };

    // parse all CLI args now, so that -hf is available below for remote preset resolution
    parse_cli_args();

    // HF_TOKEN / -hft win; otherwise fall back to the huggingface_hub token file
    if (params.hf_token.empty()) {
        params.hf_token = common_hf_token_from_files();
    }

    postprocess_cpu_params(params.cpuparams,       nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    // upstream's params.speculative.draft.* spelling (set by a caller before parsing) folds into the flat fields every reader uses
    common_params_speculative_fold_draft(params.speculative);

    postprocess_cpu_params(params.speculative.cpuparams,       &params.cpuparams);
    postprocess_cpu_params(params.speculative.cpuparams_batch, &params.cpuparams_batch);

    // Ollama compatibility (default OFF): an explicit fork flag/env always wins over an OLLAMA_* variable or a store lookup.
    {
        auto explicit_set = [&](std::initializer_list<const char *> names, const char * llama_env) {
            for (int i = 1; i < argc; i++) {
                for (const char * n : names) { if (std::string(argv[i]) == n) return true; }
            }
            return llama_env && std::getenv(llama_env) != nullptr;
        };
        if (params.ollama_env) {
            ollama_util::env_overrides eo = ollama_util::read_env([](const char * k) { return std::getenv(k); });
            for (auto & w : eo.warnings) { LOG_WRN("--ollama-env: %s\n", w.c_str()); }
            if (eo.has_kv && !explicit_set({"-ctk", "--cache-type-k"}, "LLAMA_ARG_CACHE_TYPE_K")) { params.cache_type_k = eo.kv_type; }
            if (eo.has_kv && !explicit_set({"-ctv", "--cache-type-v"}, "LLAMA_ARG_CACHE_TYPE_V")) { params.cache_type_v = eo.kv_type; }
            if (eo.has_flash && !explicit_set({"-fa", "--flash-attn"}, "LLAMA_ARG_FLASH_ATTN")) { params.flash_attn = eo.flash; }
            if (eo.has_ctx && !explicit_set({"-c", "--ctx-size"}, "LLAMA_ARG_CTX_SIZE")) { params.n_ctx = (int32_t) eo.ctx; }
            if (eo.has_parallel && !explicit_set({"-np", "--parallel"}, "LLAMA_ARG_N_PARALLEL")) { params.n_parallel = (int32_t) eo.parallel; }
            if (eo.has_max_models && !explicit_set({"--models-max"}, "LLAMA_ARG_MODELS_MAX")) { params.models_max = (int) eo.max_models; }
            if (!explicit_set({"--models-keep-alive"}, "LLAMA_ARG_MODELS_KEEP_ALIVE")) {
                // ollama's default is 5 minutes; an unparseable OLLAMA_KEEP_ALIVE falls back to it (config.go:126-143)
                int64_t ms = 5 * 60 * 1000;
                if (eo.has_keep_alive && !ollama_util::keep_alive_ms(eo.keep_alive, ms)) {
                    LOG_WRN("--ollama-env: OLLAMA_KEEP_ALIVE=%s is not a duration, using ollama's default of 5m\n", eo.keep_alive.c_str());
                    ms = 5 * 60 * 1000;
                }
                params.models_keep_alive_ms = ms;
            }
            LOG_INF("--ollama-env: kv=%s flash_attn=%d n_ctx=%d n_parallel=%d models_keep_alive_ms=%lld\n",
                    eo.has_kv ? eo.kv_type.c_str() : "-", (int) params.flash_attn, (int) params.n_ctx, (int) params.n_parallel,
                    (long long) params.models_keep_alive_ms);
        }
        if (!params.ollama_model.empty()) {
            if (explicit_set({"-m", "--model"}, "LLAMA_ARG_MODEL")) {
                throw std::invalid_argument("error: --ollama-model and --model are mutually exclusive\n");
            }
            const std::string dir = !params.ollama_models_dir.empty() ? params.ollama_models_dir : ollama_util::store_default_dir();
            ollama_util::store_model sm;
            std::string err;
            if (!ollama_util::store_resolve(dir, params.ollama_model, sm, err)) {
                throw std::invalid_argument("error: --ollama-model: " + err + "\n");
            }
            params.model.path = sm.model_blob;
            if (!sm.projector_blob.empty() && params.mmproj.path.empty() && !params.no_mmproj) { params.mmproj.path = sm.projector_blob; }
            // Ollama's newer blobs keep the vision tower inline (v.*) and have no projector layer. For the arches the clip
            // loader can read from such a blob (tools/mtmd/clip.cpp apply_ollama_qwen3vl_compat / _qwen25vl_compat) the
            // blob is also the mmproj, as Ollama does with its own allowlist.
            // SOURCE-READ: tmp/arex1014/engq_a_1019/q25vl_1020/_ol_llama_server.go.txt (ollama a8aaf9f llm/llama_server.go, MIT) portability-ok: provenance of the read, not a runtime path
            // lines 863-893; re-specified, no code copied.
            // SOURCE-NUANCE: ollama gates on THREE things -- no projector layer, at least one "v." tensor, and the arch on an
            // explicit allowlist that mirrors its compat clip handlers -- because un-translated Ollama vision tensors make the
            // clip loader abort the whole model load; an arch outside the list is served text-only. Our list is the arches
            // our clip.cpp translates (qwen3vl, qwen3vlmoe, qwen25vl), not ollama's twelve.
            bool mmproj_from_blob = false;
            if (sm.projector_blob.empty() && params.mmproj.path.empty() && !params.no_mmproj) {
                static const char * const clip_compat_arches[] = { "qwen3vl", "qwen3vlmoe", "qwen25vl" };
                struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
                if (gguf_context * g = gguf_init_from_file(sm.model_blob.c_str(), gp)) {
                    const int64_t aid = gguf_find_key(g, "general.architecture");
                    const std::string arch = aid >= 0 && gguf_get_kv_type(g, aid) == GGUF_TYPE_STRING ? gguf_get_val_str(g, aid) : "";
                    bool has_vision = false;
                    for (int64_t i = 0; i < gguf_get_n_tensors(g) && !has_vision; ++i) {
                        has_vision = std::strncmp(gguf_get_tensor_name(g, i), "v.", 2) == 0;
                    }
                    gguf_free(g);
                    const bool known = std::find_if(std::begin(clip_compat_arches), std::end(clip_compat_arches),
                                                    [&](const char * a) { return arch == a; }) != std::end(clip_compat_arches);
                    if (has_vision && known) {
                        params.mmproj.path = sm.model_blob;
                        mmproj_from_blob   = true;
                    } else if (has_vision) {
                        LOG_WRN("--ollama-model: %s (%s) has an inline vision tower that the clip loader cannot read yet; serving text only\n",
                                params.ollama_model.c_str(), arch.c_str());
                    }
                }
            }
            for (const auto & a : sm.adapter_blobs) { params.lora_adapters.push_back({ a, 1.0, "", "", nullptr }); }
            if (!sm.model_shards.empty()) {
                LOG_WRN("--ollama-model: %s has %d extra model layer(s) (split GGUF); only the first is loaded\n", params.ollama_model.c_str(), (int) sm.model_shards.size());
            }
            if (params.ollama_models_dir.empty()) { params.ollama_models_dir = dir; }
            if (params.ollama_go_template && !sm.template_text.empty()) {
                // ollama prefers the GGUF's own chat template when it has one (server/images.go:850-860 PreferChatTemplate); the Go
                // TEMPLATE layer is the fallback, translated to Jinja for the fork's engine (server-ollama-gotmpl.h)
                bool gguf_has_template = false;
                struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
                if (gguf_context * g = gguf_init_from_file(sm.model_blob.c_str(), gp)) {
                    gguf_has_template = gguf_find_key(g, "tokenizer.chat_template") >= 0;
                    gguf_free(g);
                }
                if (gguf_has_template) {
                    LOG_INF("--ollama-go-template: the GGUF has its own chat template; the Go TEMPLATE layer is not used\n");
                } else if (!params.chat_template.empty()) {
                    LOG_INF("--ollama-go-template: --chat-template was given; the Go TEMPLATE layer is not used\n");
                } else {
                    std::string jinja, terr;
                    if (ollama_gotmpl::translate(sm.template_text, jinja, terr)) {
                        params.chat_template = jinja;
                        params.use_jinja = true;
                        LOG_INF("--ollama-go-template: translated the Go TEMPLATE layer (%d bytes) to Jinja (%d bytes) and set it as the chat template\n",
                                (int) sm.template_text.size(), (int) jinja.size());
                    } else {
                        LOG_WRN("--ollama-go-template: cannot translate the Go TEMPLATE layer (%s); the model keeps no chat template\n", terr.c_str());
                    }
                }
            }
            // evidence line: always on
            LOG_INF("--ollama-model: %s -> %s (mmproj=%s, adapters=%d, template=%s)\n", params.ollama_model.c_str(), sm.model_blob.c_str(),
                    mmproj_from_blob ? "the model blob (inline vision tower)" : sm.projector_blob.empty() ? "none" : sm.projector_blob.c_str(),
                    (int) sm.adapter_blobs.size(), sm.template_text.empty() ? "no" : "yes");
        }
    }

    // default the mmproj device to the global device selection if not set explicitly with -mmdev
    // (the fork keeps --device as a string; its parsed list ends with a nullptr, so a "none" list yields nullptr = CPU, as upstream)
    if (params.mmproj_use_gpu && params.mmproj_device == nullptr && !params.devices.empty()) {
        const auto devs = parse_device_list(params.devices);
        params.mmproj_device = devs.empty() ? nullptr : devs.front();
        params.mmproj_use_gpu = params.mmproj_device != nullptr;
    }

    if (params.prompt_cache_all && (params.interactive || params.interactive_first)) {
        throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
    }

    // Fade and slope both reshape the same penalty window; the sampler would apply only the fade.
    if (params.sampling.penalty_decay > 0 && params.sampling.penalty_slope > 0.0f && params.sampling.penalty_slope < 1.0f) {
        throw std::invalid_argument("error: --repeat-decay cannot be combined with --repeat-penalty-slope < 1\n");
    }
    // Skew acts in the plain temperature draw only; these draw elsewhere, so the setting would be silently dropped.
    if (params.sampling.skew != 0.0f &&
        (params.sampling.mirostat != 0 || params.sampling.adaptive_target >= 0.0f || params.sampling.fsd_alpha >= 0.0f ||
         (params.sampling.watermark && params.sampling.watermark_scheme != common_watermark::SCHEME_KGW))) {
        throw std::invalid_argument("error: --skew cannot be combined with --mirostat, --adaptive-target, --fsd-alpha or a "
                                    "gumbel/synthid --watermark (they make their own draw)\n");
    }
    // --watermark with no key would sample unmarked text while claiming otherwise.
    if (params.sampling.watermark && !params.sampling.watermark_key_set) {
        throw std::invalid_argument("error: --watermark needs --watermark-key (or LLAMA_ARG_WATERMARK_KEY)\n");
    }
    // ...and mirostat / adaptive-p draw in branches no watermark scheme reaches.
    if (params.sampling.watermark && (params.sampling.mirostat != 0 || params.sampling.adaptive_target >= 0.0f)) {
        throw std::invalid_argument("error: --watermark cannot be combined with --mirostat or --adaptive-target\n");
    }
    // --adaptive-p-untruncated refuse: the server-wide defaults decide here; a request can still override them.
    if (ctx_arg.ex != LLAMA_EXAMPLE_SERVER) {
        const std::string refusal = common_sampling_adaptive_p_refusal(params.sampling);
        if (!refusal.empty()) {
            throw std::invalid_argument("error: " + refusal + "\n");
        }
    }

    const bool skip_model_download =
        // server will call common_params_handle_models() later, so we skip it here
        ctx_arg.ex == LLAMA_EXAMPLE_SERVER ||
        // download calls common_params_handle_models() itself and prints the paths
        ctx_arg.ex == LLAMA_EXAMPLE_DOWNLOAD ||
        // export_graph_ops loads only metadata
        ctx_arg.ex == LLAMA_EXAMPLE_EXPORT_GRAPH_OPS;

    if (!skip_model_download) {
        // handle model and download
        common_models_handler handler = common_models_handler_init(params, ctx_arg.ex);
        common_models_handler_apply(handler, params);

        // model is required (except for server)
        // TODO @ngxson : maybe show a list of available models in CLI in this case
        bool can_skip_model = params.usage || params.completion || !params.server_base.empty();
        if (!can_skip_model && params.model.empty()) {
            throw std::invalid_argument("error: --model is required\n");
        }
    }

    if (params.escape) {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        for (auto & antiprompt : params.antiprompt) {
            string_process_escapes(antiprompt);
        }
        for (auto & seq_breaker : params.sampling.dry_sequence_breakers) {
            string_process_escapes(seq_breaker);
        }
    }

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    const bool mcp_enabled = !params.mcp_servers_config.empty() || !params.mcp_servers_json.empty();
    if ((!params.server_tools.empty() || mcp_enabled) && !params.cors_origins_explicit) {
        LOG_WRN("server tools or MCP servers are enabled, using localhost as default CORS origin (change via --cors-origins)\n");
        params.cors_origins = "localhost";
    }

    // pad tensor_buft_overrides for llama_params_fit:
    const size_t ntbo = llama_max_tensor_buft_overrides();
    while (params.tensor_buft_overrides.size() < ntbo) {
        params.tensor_buft_overrides.push_back({nullptr, nullptr});
    }

    if (!params.speculative.tensor_buft_overrides.empty()) {
        params.speculative.tensor_buft_overrides.push_back({nullptr, nullptr});
    }

    if (!params.chat_template.empty() && !common_chat_verify_template(params.chat_template, params.use_jinja)) {
        throw std::runtime_error(string_format(
            "error: the supplied chat template is not supported: %s%s\n",
            params.chat_template.c_str(),
            params.use_jinja ? "" : "\nnote: llama.cpp was started without --jinja, we only support commonly used templates"
        ));
    }

    // if the preserve_reasoning kwarg was not specified explicitly, enable it by default
    if (!params.default_template_kwargs.count("preserve_reasoning")) {
        params.default_template_kwargs["preserve_reasoning"] = "true";
    }

    return true;
}

static void common_params_print_usage(common_params_context & ctx_arg) {
    auto print_options = [](std::vector<common_arg *> & options) {
        for (common_arg * opt : options) {
            printf("%s", opt->to_string().c_str());
        }
    };

    std::vector<common_arg *> common_options;
    std::vector<common_arg *> sampling_options;
    std::vector<common_arg *> spec_options;
    std::vector<common_arg *> specific_options;
    for (auto & opt : ctx_arg.options) {
        // in case multiple LLAMA_EXAMPLE_* are set, we prioritize the LLAMA_EXAMPLE_* matching current example
        if (opt.is_sampling) {
            sampling_options.push_back(&opt);
        } else if (opt.is_spec) {
            spec_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }
    bool first = true;
    auto print_section = [&](const char * header, std::vector<common_arg *> & options) {
        if (options.empty()) {
            return;
        }
        printf("%s----- %s -----\n\n", first ? "" : "\n\n", header);
        first = false;
        print_options(options);
    };
    print_section("common params",           common_options);
    print_section("sampling params",         sampling_options);
    print_section("speculative params",      spec_options);
    print_section("example-specific params", specific_options);
}

static void common_params_print_completion(common_params_context & ctx_arg) {
    std::vector<common_arg *> common_options;
    std::vector<common_arg *> sampling_options;
    std::vector<common_arg *> spec_options;
    std::vector<common_arg *> specific_options;

    for (auto & opt : ctx_arg.options) {
        if (opt.is_sampling) {
            sampling_options.push_back(&opt);
        } else if (opt.is_spec) {
            spec_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }

    printf("_llama_completions() {\n");
    printf("    local cur prev opts\n");
    printf("    COMPREPLY=()\n");
    printf("    cur=\"${COMP_WORDS[COMP_CWORD]}\"\n");
    printf("    prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n\n");

    printf("    opts=\"");
    auto print_options = [](const std::vector<common_arg *> & options) {
        for (const common_arg * opt : options) {
            for (const char * arg : opt->args) {
                printf("%s ", arg);
            }
        }
    };

    print_options(common_options);
    print_options(sampling_options);
    print_options(spec_options);
    print_options(specific_options);
    printf("\"\n\n");

    printf("    case \"$prev\" in\n");
    printf("        --model|-m)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.gguf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        --grammar-file)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.gbnf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        --chat-template-file)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.jinja' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        *)\n");
    printf("            COMPREPLY=( $(compgen -W \"${opts}\" -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("    esac\n");
    printf("}\n\n");

    std::set<std::string> executables = {
        "llama-batched",
        "llama-batched-bench",
        "llama-bench",
        "llama-cli",
        "llama-completion",
        "llama-convert-llama2c-to-ggml",
        "llama-cvector-generator",
        "llama-debug",
        "llama-diffusion-cli",
        "llama-embedding",
        "llama-eval-callback",
        "llama-export-lora",
        "llama-finetune",
        "llama-fit-params",
        "llama-gemma3-cli",
        "llama-gen-docs",
        "llama-gguf",
        "llama-gguf-hash",
        "llama-gguf-split",
        "llama-idle",
        "llama-imatrix",
        "llama-llava-cli",
        "llama-lookahead",
        "llama-lookup",
        "llama-lookup-create",
        "llama-lookup-merge",
        "llama-lookup-stats",
        "llama-minicpmv-cli",
        "llama-mtmd-cli",
        "llama-parallel",
        "llama-passkey",
        "llama-perplexity",
        "llama-q8dot",
        "llama-quantize",
        "llama-qwen2vl-cli",
        "llama-retrieval",
        "llama-save-load-state",
        "llama-server",
        "llama-simple",
        "llama-simple-chat",
        "llama-speculative",
        "llama-speculative-simple",
        "llama-tokenize",
        "llama-tts",
        "llama-vdot"
    };

    for (const auto& exe : executables) {
        printf("complete -F _llama_completions %s\n", exe.c_str());
    }
}

static std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    auto dev_names = string_split<std::string>(value, ',');
    if (dev_names.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);
    } else {
        ggml_backend_load_all();
        for (const auto & device : dev_names) {
            auto * dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                throw std::invalid_argument(string_format("invalid device: %s", device.c_str()));
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}

void common_print_available_devices() {
    constexpr size_t MiB = 1024 * 1024;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            devices.push_back(dev);
        }
    }
    printf("Available devices:\n");

    if (devices.empty()) {
        printf("  (none)\n");
        return;
    }
    for (auto * dev : devices) {
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / MiB, free / MiB);
    }
}

static void add_rpc_devices(const std::string & servers) {
    auto rpc_servers = string_split<std::string>(servers, ',');
    if (rpc_servers.empty()) {
        throw std::invalid_argument("no RPC servers specified");
    }
    ggml_backend_load_all();
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        throw std::invalid_argument("failed to find RPC backend");
    }
    typedef ggml_backend_reg_t (*ggml_backend_rpc_add_server_t)(const char * endpoint);
    ggml_backend_rpc_add_server_t ggml_backend_rpc_add_server_fn = (ggml_backend_rpc_add_server_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!ggml_backend_rpc_add_server_fn) {
        throw std::invalid_argument("failed to find RPC add server function");
    }
    for (const auto & server : rpc_servers) {
        auto reg = ggml_backend_rpc_add_server_fn(server.c_str());
        // nullptr = unreachable, zero devices, or a refused HELLO (e.g. an upstream peer: this fork speaks
        // RPC major 133). Registering it would hand a null reg to the registry and crash later at
        // GGML_ASSERT(buft) in model load, so fail here with the endpoint named.
        if (!reg) {
            throw std::invalid_argument("failed to register RPC server " + server +
                                        " (unreachable, no devices, or protocol version mismatch -- see log above)");
        }
        ggml_backend_register(reg);
    }
}

bool common_params_to_map(int argc, char ** argv, llama_example ex, std::map<common_arg, std::string> & out_map) {
    common_params dummy_params;
    common_params_context ctx_arg = common_params_parser_init(dummy_params, ex, nullptr);

    std::unordered_map<std::string, common_arg *> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = &opt;
        }
        for (const auto & arg : opt.args_neg) {
            arg_to_options[arg] = &opt;
        }
    }

    // TODO @ngxson : find a way to deduplicate this code

    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };

    std::set<std::string> seen_args;

    for (int i = 1; i < argc; i++) {
        const std::string arg_prefix = "--";

        std::string arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }
        if (arg_to_options.find(arg) == arg_to_options.end()) {
            throw std::invalid_argument(string_format("error: invalid argument: %s", arg.c_str()));
        }
        if (!seen_args.insert(arg).second) {
            const bool skip = (arg == "--spec-type");

            if (!skip) {
                LOG_WRN("DEPRECATED: argument '%s' specified multiple times, use comma-separated values instead (only last value will be used)\n", arg.c_str());
            }
        }
        auto opt = *arg_to_options[arg];
        std::string val;
        if (opt.value_hint == nullptr && opt.value_hint_2 == nullptr) {
            // bool arg (need to reverse the meaning for negative args)
            bool is_neg = std::find(opt.args_neg.begin(), opt.args_neg.end(), arg) != opt.args_neg.end();
            val = is_neg ? "0" : "1";
        }
        if (opt.value_hint != nullptr) {
            // arg with single value
            check_arg(i);
            val = argv[++i];
        }
        if (opt.value_hint_2 != nullptr) {
            // TODO: support arg with 2 values
            throw std::invalid_argument("error: argument with 2 values is not yet supported\n");
        }
        out_map[opt] = val;
    }

    return true;
}

#ifdef _WIN32
struct utf8_argv {
    std::vector<std::string> buf;
    std::vector<std::string> acp;
    std::vector<char*> ptrs;
};

static utf8_argv make_utf8_argv() {
    utf8_argv out;
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv) return out;

    out.buf.reserve(wargc);
    for (int i = 0; i < wargc; ++i) {
        int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        if (n <= 0) { out.buf.emplace_back(); continue; }
        auto& s = out.buf.emplace_back();
        s.resize(static_cast<size_t>(n - 1));
        (void)WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), n, nullptr, nullptr);
    }
    // the same arguments in the ANSI code page, as the CRT produced argv; used only to recognise
    // an argv that is still the process command line
    out.acp.reserve(wargc);
    for (int i = 0; i < wargc; ++i) {
        int n = WideCharToMultiByte(CP_ACP, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        auto& s = out.acp.emplace_back();
        if (n <= 0) { continue; }
        s.resize(static_cast<size_t>(n - 1));
        (void)WideCharToMultiByte(CP_ACP, 0, wargv[i], -1, s.data(), n, nullptr, nullptr);
    }
    LocalFree(wargv);

    out.ptrs.reserve(out.buf.size() + 1);
    for (auto& s : out.buf) out.ptrs.push_back(s.data());
    out.ptrs.push_back(nullptr);
    return out;
}
#endif

bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
#ifdef _WIN32
    auto utf8 = make_utf8_argv();
    // repair argv only when it matches the process command line. A matching COUNT is not enough:
    // a tool that rewrites its argv before calling here (llama-tts-outetts maps --out-file to
    // --output, one entry for one) keeps the count, and the swap silently restored the raw
    // command line, so the rewritten flag was refused. Every pure-ASCII entry must also be
    // byte-equal; only entries carrying non-ASCII bytes (the ones the ANSI code page mangles)
    // may differ.
    bool same_command_line = static_cast<int>(utf8.buf.size()) == argc && utf8.acp.size() == utf8.buf.size();
    for (int i = 0; same_command_line && i < argc; ++i) {
        // the CRT built argv[i] from the same wide argument through the ANSI code page, so an
        // unmodified entry equals that conversion (non-ASCII text arrives as '?' or best-fit
        // characters, which is what the repair exists for) or, under a UTF-8 code page, the UTF-8 form.
        if (utf8.acp[i] != argv[i] && utf8.buf[i] != argv[i]) {
            same_command_line = false;
        }
    }
    if (same_command_line) {
        argv = utf8.ptrs.data();
    }
#endif

    auto ctx_arg = common_params_parser_init(params, ex, print_usage);
    const common_params params_org = ctx_arg.params; // the example can modify the default params

    try {
        if (!common_params_parse_ex(argc, argv, ctx_arg)) {
            ctx_arg.params = params_org;
            return false;
        }
        if (ctx_arg.params.usage) {
            common_params_print_usage(ctx_arg);
            if (ctx_arg.print_usage) {
                ctx_arg.print_usage(argc, argv);
            }
            common_log_flush(common_log_main());
            exit(0);
        }
        if (ctx_arg.params.completion) {
            common_params_print_completion(ctx_arg);
            exit(0);
        }
        params.lr.init();
        // ik main (cea2d61ff) gpt_params_parse tail: resolve the CPU affinity once, so the list stays alive until
        // the context is created (the auto-merge had dropped this line into the Windows set_process_priority).
        params.cpu_affinity = cpu_affinity_resolve(params.cpu_affinity, params.cpu_affinity_auto);
    } catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        ctx_arg.params = params_org;
        return false;
    } catch (std::exception & ex) {
        fprintf(stderr, "%s\n", ex.what());
        exit(1); // for other exceptions, we exit with status code 1
    }

    return true;
}

static std::string list_builtin_chat_templates() {
    std::vector<const char *> supported_tmpl;
    int32_t res = llama_chat_builtin_templates(nullptr, 0);
    supported_tmpl.resize(res);
    res = llama_chat_builtin_templates(supported_tmpl.data(), supported_tmpl.size());
    std::ostringstream msg;
    for (auto & tmpl : supported_tmpl) {
        msg << tmpl << (&tmpl == &supported_tmpl.back() ? "" : ", ");
    }
    return msg.str();
}

bool common_arg_utils::is_truthy(const std::string & value) {
    return value == "on" || value == "enabled" || value == "true" || value == "1";
}

bool common_arg_utils::is_falsey(const std::string & value) {
    return value == "off" || value == "disabled" || value == "false" || value == "0";
}

bool common_arg_utils::is_autoy(const std::string & value) {
    return value == "auto" || value == "-1";
}

// Simple CSV parser that handles quoted fields and escaped quotes
// example:
//    input:  value1,"value, with, commas","value with ""escaped"" quotes",value4
//    output: [value1] [value, with, commas] [value with "escaped" quotes] [value4]
// reads one key per line (# starts a comment line, surrounding blanks and CR are stripped: a CRLF key file used to leave
// the CR on every key, which then never matched a header). Same rules as server_auth_read_key_file in tools/server.
static void read_api_key_file(const std::string & path, std::vector<std::string> & out) {
    std::ifstream key_file(path);
    if (!key_file) {
        throw std::runtime_error(string_format("error: failed to open file '%s'\n", path.c_str()));
    }
    std::string line;
    while (std::getline(key_file, line)) {
        size_t a = 0, b = line.size();
        while (a < b && (line[a] == ' ' || line[a] == '\t' || line[a] == '\r')) a++;
        while (b > a && (line[b - 1] == ' ' || line[b - 1] == '\t' || line[b - 1] == '\r')) b--;
        if (b > a && line[a] != '#') {
            out.push_back(line.substr(a, b - a));
        }
    }
}

static std::vector<std::string> parse_csv_row(const std::string& input) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (size_t i = 0; i < input.length(); ++i) {
        char ch = input[i];

        if (ch == '"') {
            if (!in_quotes) {
                // start of quoted field (only valid if at beginning of field)
                if (!field.empty()) {
                    // quote appeared in middle of unquoted field, treat as literal
                    field += '"';
                } else {
                    in_quotes = true; // start
                }
            } else {
                if (i + 1 < input.length() && input[i + 1] == '"') {
                    // escaped quote: ""
                    field += '"';
                    ++i; // skip the next quote
                } else {
                    in_quotes = false; // end
                }
            }
        } else if (ch == ',') {
            if (in_quotes) {
                field += ',';
            } else {
                fields.push_back(std::move(field));
                field.clear();
            }
        } else {
            field += ch;
        }
    }

    // Add the last field
    fields.push_back(std::move(field));

    return fields;
}

common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    // per-example default params
    // we define here to make sure it's included in llama-gen-docs
    if (ex == LLAMA_EXAMPLE_COMPLETION) {
        params.use_jinja = false;   // disable jinja by default
    } else if (ex == LLAMA_EXAMPLE_MTMD) {
        params.use_jinja = false;   // disable jinja by default
        params.sampling.temp = 0.2; // lower temp by default for better quality
    } else if (ex == LLAMA_EXAMPLE_SERVER) {
        params.n_parallel = -1;     // auto by default
    } else if (ex == LLAMA_EXAMPLE_TOKENIZE) {
        params.parse_special = true; // parse special tokens by default, like the old tokenize tool
    } else if (ex == LLAMA_EXAMPLE_TTS) {
        params.out_file = "output.wav";
        params.sampling.penalty_repeat = 1.05f;
        params.sampling.penalty_last_n = -1;
    }

    params.use_color = tty_can_use_colors();

    common_params_context ctx_arg(params);
    ctx_arg.print_usage = print_usage;
    ctx_arg.ex          = ex;

    std::string sampler_type_chars;
    std::string sampler_type_names;
    // ADAPTED 2026-09-13: this fork's sampler chain is driven by `samplers_sequence`
    // (std::vector<llama_sampler_type>), NOT by upstream's `samplers`
    // (std::vector<common_sampler_type>). samplers_sequence is the field the chain builder
    // actually reads -- sampling.cpp:152, :325, :454, :1511 -- so writing upstream's field
    // would have configured nothing even if it had existed.
    //
    // `llama_sampler_type` is `enum class ... : char` whose enumerator VALUE IS the chr
    // (sampling.h:15-27, DRY='d', TOP_K='k', ...), so the cast is the fork's own to_chr.
    for (const auto & sampler : params.sampling.samplers_sequence) {
        sampler_type_chars += static_cast<char>(sampler);
        sampler_type_names += llama_sampling_type_to_str(sampler) + ";";
    }
    if (!sampler_type_names.empty()) {
        sampler_type_names.pop_back(); // remove last semicolon
    }

    /**
     * filter options by example
     * rules:
     * - all examples inherit options from LLAMA_EXAMPLE_COMMON
     * - if LLAMA_EXAMPLE_* is set (other than COMMON), we only show the option in the corresponding example
     * - if both {LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_*,} are set, we will prioritize the LLAMA_EXAMPLE_* matching current example
     */
    auto add_opt = [&](common_arg arg) {
        // download only exposes the handful of args explicitly tagged for it
        const bool inherit_common = ex != LLAMA_EXAMPLE_DOWNLOAD;
        if ((arg.in_example(ex) || (inherit_common && arg.in_example(LLAMA_EXAMPLE_COMMON))) && !arg.is_exclude(ex)) {
            ctx_arg.options.push_back(std::move(arg));
        }
    };

    add_opt(common_arg(
        {"-h", "--help", "--usage"},
        "print usage and exit",
        [](common_params & params) {
            params.usage = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD}));
    add_opt(common_arg(
        {"--version"},
        "show version and build info",
        [](common_params &) {
            llama_print_build_info(llama_version());
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"-cl", "--cache-list"},
        "show list of models in cache",
        [](common_params &) {
            auto models = common_list_cached_models();
            printf("number of models in cache: %zu\n", models.size());
            for (size_t i = 0; i < models.size(); i++) {
                printf("%4zu. %s\n", i + 1, models[i].to_string().c_str());
            }
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"--completion-bash"},
        "print source-able bash completion script for llama.cpp",
        [](common_params & params) {
            params.completion = true;
        }
    ));
    add_opt(common_arg(
        {"--server-base"}, "URL",
        string_format("connect to this server instead of starting a new one, example: 'http://localhost:9931' (default: none)"),
        [](common_params & params, const std::string & value) {
            params.server_base = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--verbose-prompt"},
        string_format("print a verbose prompt before generation (default: %s)", params.verbose_prompt ? "true" : "false"),
        [](common_params & params) {
            params.verbose_prompt = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--display-prompt"},
        {"--no-display-prompt"},
        string_format("whether to print prompt at generation (default: %s)", params.display_prompt ? "true" : "false"),
        [](common_params & params, bool value) {
            params.display_prompt = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-co", "--color"}, "[on|off|auto]",
        "Colorize output to distinguish prompt and user input from generations ('on', 'off', or 'auto', default: 'auto')\n"
        "'auto' enables colors when output is to a terminal",
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.use_color = true;
            } else if (is_falsey(value)) {
                params.use_color = false;
            } else if (is_autoy(value)) {
                params.use_color = tty_can_use_colors();
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --color: '%s'\n", value.c_str()));
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP}));
    add_opt(common_arg(
        {"-t", "--threads"}, "N",
        string_format("number of CPU threads to use during generation (default: %d)", params.cpuparams.n_threads),
        [](common_params & params, int value) {
            params.cpuparams.n_threads = value;
            if (params.cpuparams.n_threads <= 0) {
                params.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_env("LLAMA_ARG_THREADS"));
    add_opt(common_arg(
        {"-tb", "--threads-batch"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads)",
        [](common_params & params, int value) {
            params.cpuparams_batch.n_threads = value;
            if (params.cpuparams_batch.n_threads <= 0) {
                params.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ));
    add_opt(common_arg(
        {"-C", "--cpu-mask"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")",
        [](common_params & params, const std::string & mask) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Cr", "--cpu-range"}, "lo-hi",
        "range of CPUs for affinity. Complements --cpu-mask",
        [](common_params & params, const std::string & range) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    // Opt-in flush-to-zero / denormals-are-zero for the ggml CPU workers (idea: NCNN flush_denormals). Default OFF.
    add_opt(common_arg(
        {"--cpu-flush-denormals"},
        "treat denormal floats as zero in CPU compute threads (x86 MXCSR FTZ/DAZ, AArch64 FPCR.FZ); changes numerics only in the denormal range (default: off)",
        [](common_params & params) {
            params.cpuparams.flush_denormals = true;
        }
    ).set_env("LLAMA_ARG_CPU_FLUSH_DENORMALS"));
    add_opt(common_arg(
        {"--cpu-strict"}, "<0|1>",
        string_format("use strict CPU placement (default: %u)\n", (unsigned) params.cpuparams.strict_cpu),
        [](common_params & params, const std::string & value) {
            params.cpuparams.strict_cpu = std::stoul(value);
        }
    ));
    // Opt-in physical-core placement (Thireus/ik_llama #2565's auto mode, every physical core, Windows + Linux). Spelled
    // --cpu-placement so it cannot collide with ik's --cpu-affinity / -cm / -cr (--cpu-affinity-mask/-range), which the ik
    // main trial branch adds as a separate per-thread pinning list. Resolved in postprocess_cpu_params; -C/-Cr win.
    add_opt(common_arg(
        {"--cpu-placement"}, "{none,physical}",
        "CPU worker placement when no -C/-Cr mask is given (default: none)\n"
        "physical: one worker per physical core (lowest logical CPU of each core, SMT siblings unused), strict placement;\n"
        "applies to --threads and --threads-batch. An explicit -C/-Cr (or -Cb/-Crb) mask wins.",
        [](common_params & params, const std::string & value) {
            bool physical;
            if (!common_cpu_placement_parse(value, physical)) {
                throw std::invalid_argument("invalid --cpu-placement value '" + value + "' (expected none or physical)");
            }
            params.cpuparams.place_physical       = physical;
            params.cpuparams_batch.place_physical = physical;
        }
    ));
    // ik #2565 / #2588 explicit worker pinning (ggml-cpu.h cpu_affinity: worker ith -> cpu_affinity[ith % n]).
    // It sits beside the threadpool cpumask above rather than replacing it. ik spells the long names
    // --cpu-mask / --cpu-range, which this tree already gives to -C / -Cr. ik's short names -cm / -cr do
    // not collide (arg names are case-sensitive), so they keep ik's spelling, and the long names become
    // --cpu-affinity-mask / --cpu-affinity-range. Resolved to params.cpu_affinity after parsing
    // (cpu_affinity_resolve, above). Also parsed inside --draft-params (common_speculative_apply_draft_affinity).
    add_opt(common_arg(
        {"--cpu-affinity"},
        "pin CPU workers to the physical P-cores (hybrid CPUs only; in --draft-params: the E-cores)",
        [](common_params & params) {
            params.cpu_affinity.clear();
            params.cpu_affinity_auto       = true;
            params.cpu_affinity_configured = true;
        }
    ));
    add_opt(common_arg(
        {"-cm", "--cpu-affinity-mask"}, "MASK",
        "pin CPU workers to the logical CPUs set in MASK (hex or decimal bitmask, e.g. 0x55; 64 CPUs max, use -cr beyond)",
        [](common_params & params, const std::string & value) {
            if (!cpu_affinity_parse_mask(value, params.cpu_affinity)) {
                throw std::invalid_argument("invalid CPU affinity mask: " + value);
            }
            params.cpu_affinity_auto       = false;
            params.cpu_affinity_configured = true;
        }
    ));
    add_opt(common_arg(
        {"-cr", "--cpu-affinity-range"}, "LIST",
        "pin CPU workers to the given logical CPUs (e.g. 0-3,8,10-11)",
        [](common_params & params, const std::string & value) {
            if (!cpu_affinity_parse_range(value, params.cpu_affinity)) {
                throw std::invalid_argument("invalid CPU affinity range: " + value);
            }
            params.cpu_affinity_auto       = false;
            params.cpu_affinity_configured = true;
        }
    ));
    add_opt(common_arg(
        {"--prio"}, "N",
        string_format("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: %d)\n", params.cpuparams.priority),
        [](common_params & params, int prio) {
            if (prio < GGML_SCHED_PRIO_LOW || prio > GGML_SCHED_PRIO_REALTIME) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll"}, "<0...100>",
        string_format("use polling level to wait for work (0 - no polling, default: %u)\n", (unsigned) params.cpuparams.poll),
        [](common_params & params, const std::string & value) {
            params.cpuparams.poll = std::stoul(value);
        }
    ));
    add_opt(common_arg(
        {"-Cb", "--cpu-mask-batch"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Crb", "--cpu-range-batch"}, "lo-hi",
        "ranges of CPUs for affinity. Complements --cpu-mask-batch",
        [](common_params & params, const std::string & range) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(common_arg(
        {"--cpu-strict-batch"}, "<0|1>",
        "use strict CPU placement (default: same as --cpu-strict)",
        [](common_params & params, int value) {
            params.cpuparams_batch.strict_cpu = value;
        }
    ));
    add_opt(common_arg(
        {"--prio-batch"}, "N",
        string_format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.cpuparams_batch.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll-batch"}, "<0|1>",
        "use polling to wait for work (default: same as --poll)",
        [](common_params & params, int value) {
            params.cpuparams_batch.poll = value;
        }
    ));
    add_opt(common_arg(
        {"-lcs", "--lookup-cache-static"}, "FNAME",
        "path to static lookup cache to use for lookup decoding (not updated by generation)",
        [](common_params & params, const std::string & value) {
            params.speculative.lookup_cache_static = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_SPECULATIVE})); // + SPECULATIVE: llama-speculative-simple --spec-type ngram_cache reads them too
    add_opt(common_arg(
        {"-lcd", "--lookup-cache-dynamic"}, "FNAME",
        "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
        [](common_params & params, const std::string & value) {
            params.speculative.lookup_cache_dynamic = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_SPECULATIVE})); // + SPECULATIVE (written at exit, upstream 83c2d012a)
    add_opt(common_arg(
        {"-c", "--ctx-size"}, "N",
        string_format("size of the prompt context (default: %d, 0 = loaded from model)", params.n_ctx),
        [](common_params & params, int value) {
            params.n_ctx = value;
            if (value == 0) {
                // disable context reduction in llama_params_fit if the user explicitly requests the full context size:
                params.fit_params_min_ctx = UINT32_MAX;
            }
        }
    ).set_env("LLAMA_ARG_CTX_SIZE"));
    add_opt(common_arg(
        { "--kv-unified-per-slot" }, "N",
        "context limit per parallel slot (default: unset, behavior unchanged).\n"
        "when set without -c/--ctx-size, the shared KV pool is sized to n_parallel*N",
        [](common_params & params, int value) {
            params.kv_unified_per_slot = value;
        }
    ).set_env("LLAMA_ARG_KV_UNIFIED_PER_SLOT").set_examples({ LLAMA_EXAMPLE_SERVER }));
    add_opt(common_arg(
        {"-n", "--predict", "--n-predict"}, "N",
        string_format(
            ex == LLAMA_EXAMPLE_COMPLETION
                ? "number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)"
                : "number of tokens to predict (default: %d, -1 = infinity)",
            params.n_predict),
        [](common_params & params, int value) {
            params.n_predict = value;
        }
    ).set_env("LLAMA_ARG_N_PREDICT"));
    add_opt(common_arg(
        {"-b", "--batch-size"}, "N",
        string_format("logical maximum batch size (default: %d)", params.n_batch),
        [](common_params & params, int value) {
            params.n_batch = value;
        }
    ).set_env("LLAMA_ARG_BATCH"));
    add_opt(common_arg(
        {"-ub", "--ubatch-size"}, "N",
        string_format("physical maximum batch size (default: %d)", params.n_ubatch),
        [](common_params & params, int value) {
            params.n_ubatch = value;
            params.n_ubatch_explicit = true;
        }
    ).set_env("LLAMA_ARG_UBATCH"));
    add_opt(common_arg(
        {"--decision-budget"}, "auto|model|N",
        "how /v1/systemone fits a long prompt into a decision model (default: auto). auto = nothing is cut: the context and the physical batch are "
        "sized to the model (unless -c/-ub are given) and a state that still does not fit is split in overlapping windows that each carry the whole "
        "question and options; model = the head budget the model was trained with, question and options are cut to it (parity with the reference "
        "implementation); N = an explicit head budget in tokens. A request can override it with \"decision_budget\"",
        [](common_params & params, const std::string & value) {
            if (value != "auto" && value != "model" && (value.empty() || value.find_first_not_of("0123456789") != std::string::npos || std::stoul(value) == 0)) {
                throw std::invalid_argument("error: --decision-budget is auto, model or a positive number of tokens\n");
            }
            params.decision_budget = value;
        }
    ).set_env("LLAMA_ARG_DECISION_BUDGET").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--decision-window"}, "auto|N",
        "the longest prompt a decision reads in one pass (default: auto = what the context takes). N = a state is read in windows of at most N tokens "
        "even if the whole prompt would fit, the logits of the windows are combined by --decision-window-agg. A request can override it with \"decision_window\"",
        [](common_params & params, const std::string & value) {
            if (value != "auto" && (value.empty() || value.find_first_not_of("0123456789") != std::string::npos || std::stoul(value) < 64)) {
                throw std::invalid_argument("error: --decision-window is auto or a number of tokens (at least 64)\n");
            }
            params.decision_window = value;
        }
    ).set_env("LLAMA_ARG_DECISION_WINDOW").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--decision-window-agg"}, "max|mean|lse|first|maxconf|medsum",
        "how the per-option logits of the windows of a long state are combined (default: max). max = best window (MaxP), mean = average "
        "(AvgP), lse = log-sum-exp, first = first window only (a cut), maxconf = the window that is most sure of its answer, "
        "medsum = sum over the windows of the log-probabilities minus their median per option (what every window says is taken off). "
        "A request can override it with \"decision_window_agg\"",
        [](common_params & params, const std::string & value) {
            if (value != "max" && value != "mean" && value != "lse" && value != "first" && value != "maxconf" && value != "medsum") {
                throw std::invalid_argument("error: --decision-window-agg is max, mean, lse, first, maxconf or medsum\n");
            }
            params.decision_window_agg = value;
        }
    ).set_env("LLAMA_ARG_DECISION_WINDOW_AGG").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--keep"}, "N",
        string_format("number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n_keep),
        [](common_params & params, int value) {
            params.n_keep = value;
        }
    ));
    add_opt(common_arg(
        {"--swa-full"},
        string_format("use full-size SWA cache (default: %s)\n"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)", params.swa_full ? "true" : "false"),
        [](common_params & params) {
            params.swa_full = true;
        }
    ).set_env("LLAMA_ARG_SWA_FULL"));
    add_opt(common_arg(
        // "-ctx-ckpt": the fork's spelling (pre-merge common.cpp @ 35f0cac75:2896, field ctx_checkpoints_n "max number
        // of context checkpoints per slot" -- the same meaning as n_ctx_checkpoints, and that old field has no reader).
        {"-ctxcp", "-ctx-ckpt", "--ctx-checkpoints", "--swa-checkpoints"}, "N",
        string_format("max number of context checkpoints to create per slot (default: %d)"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/15293)", params.n_ctx_checkpoints),
        [](common_params & params, int value) {
            params.n_ctx_checkpoints = value;
        }
    ).set_env("LLAMA_ARG_CTX_CHECKPOINTS").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        // "-ctx-ckpt-i/--ctx-checkpoints-interval": the fork's spelling (pre-merge common.cpp @ 35f0cac75:2901, field
        // ctx_checkpoints_interval "minimum number of tokens between each context checkpoint" -- the same meaning as
        // checkpoint_min_step; that old field has no reader). Only the DEFAULT differed (512 vs 8192); an alias
        // changes no default.
        {"-cms", "-ctx-ckpt-i", "--checkpoint-min-step", "--ctx-checkpoints-interval"}, "N",
        string_format("minimum spacing between context checkpoints in tokens (default: %d, 0 = no minimum)", params.checkpoint_min_step),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("checkpoint-min-step must be non-negative");
            }
            params.checkpoint_min_step = value;
        }
    ).set_env("LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // "-ctx-ckpt-t/--ctx-checkpoints-tolerance": the fork's spelling (pre-merge common.cpp @
        // 35f0cac75, field ctx_checkpoints_tolerance "the number of tokens before the full prompt to
        // create the checkpoint" -- old server-context.cpp:4205,4860 @ 35f0cac75). Unlike -ctx-ckpt/
        // -ctx-ckpt-i this has no upstream equivalent field to alias; consumed in server-context.cpp's
        // near_prompt_end check as a floor under n_ubatch, so the default (5) never narrows the
        // existing n_ubatch-based window and only widens it when explicitly raised above n_ubatch.
        {"-ctx-ckpt-t", "--ctx-checkpoints-tolerance"}, "N",
        string_format("create a context checkpoint once this many prompt tokens remain to process, "
            "if larger than the batch size would already trigger one (default: %d)", params.ctx_checkpoints_tolerance),
        [](common_params & params, int value) {
            params.ctx_checkpoints_tolerance = value;
        }
    ).set_env("LLAMA_ARG_CTX_CHECKPOINTS_TOLERANCE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: engines/vllm/vllm/v1/core/sched/scheduler.py:606-621 (a long prefill is capped only when other requests are
        // eligible, so it cannot starve them; the cap floors at budget / eligible) and engines/exllamav2/generator/dynamic.py:982-985
        // (every active job prefills its own chunk each round). Apache-2.0 / MIT, behaviour reimplemented, no code copied.
        // Row S69-VLLM-PICK-FAIR-PREFILL-CHUNK-SHARE-2026-09-29-01.
        {"--prefill-fair-share"},
        "when several slots are still processing their prompt, split each batch's prompt tokens between them (max-min fair: a short "
        "prompt takes what it needs, the rest is shared equally) instead of letting the first slot fill the whole batch. "
        "One prefilling slot keeps the whole batch (default: off)",
        [](common_params & params) {
            params.prefill_fair_share = true;
        }
    ).set_env("LLAMA_ARG_PREFILL_FAIR_SHARE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // Row S69-ENGREV-SGLANG-AUTO-TRUNCATE-2026-10-27-01 (idea from sglang auto_truncate; no code copied).
        {"--auto-truncate"},
        "a text prompt longer than the slot context is cut to fit (the first n_keep tokens and the latest tokens are kept, the "
        "middle is dropped, room is left for n_predict) instead of failing with HTTP 400; the response then carries "
        "prompt_truncated=true. Prompts with images/audio are not cut (default: off)",
        [](common_params & params) {
            params.auto_truncate = true;
        }
    ).set_env("LLAMA_ARG_AUTO_TRUNCATE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // Row S69-FASTLLM-CANCEL-LONG-PREFILL-C9-2026-10-27-01. MEASURED (CPU, 17k-token prompt, -b 2048 -ub 512): a client that
        // disconnects mid-prefill frees the slot only after the llama_decode of the current n_batch, 6.4 s later; fastllm stops at the
        // next micro-batch. Behaviour only, no code copied (fastllm, Apache-2.0).
        {"--prefill-ubatch-chunks"},
        "a slot adds at most n_ubatch prompt tokens to one scheduler step (instead of up to n_batch), so a cancelled or disconnected "
        "request is dropped after one micro-batch and the next request starts sooner. Costs one scheduler pass per micro-batch (default: off)",
        [](common_params & params) {
            params.prefill_ubatch_chunks = true;
        }
    ).set_env("LLAMA_ARG_PREFILL_UBATCH_CHUNKS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // Row S69-SERVE-CHUNKED-PREFILL-DECODE-BUDGET-AND-SINGLE-GPU-PD-MULTIPLEX-2026-11-06-01. Paper SARATHI-Serve (arXiv 2403.02310), read in full:
        // tmp/chunkpd_1106/src/2403.02310.txt:904-1050 (stall-free batching: decodes first, the leftover per-iteration token budget
        // goes to prefill chunks; budget from offline profiling against the TBT SLO; tile quantization, chunk 257 vs 256 costs +32%).
        // NO-NUANCE-FOUND: paper only, no engine code; behaviour reimplemented.
        {"--prefill-step-budget"}, "N",
        "stall-free batching: in a scheduler step that also carries decode tokens, the slots together add at most N prompt tokens, so one long "
        "prompt cannot stretch every running stream's inter-token gap to a whole n_batch decode. A step with no decode tokens keeps the whole batch. "
        "With --prefill-tbt-target-ms this is only the value used before the first measurement (default: 0 = off)",
        [](common_params & params, int value) {
            params.prefill_step_budget = std::max(0, value);
        }
    ).set_env("LLAMA_ARG_PREFILL_STEP_BUDGET").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--prefill-tbt-target-ms"}, "MS",
        "adaptive --prefill-step-budget: measure the prompt cost per token and the decode-step time on this server and size the prompt share of "
        "each mixed step so the step lasts about MS milliseconds (rounded down to 32 tokens, at least 32, at most n_batch). Needs no profiling run "
        "(SARATHI-Serve profiles offline) and follows the attention cost growing with context depth (default: 0 = off)",
        [](common_params & params, int value) {
            params.prefill_tbt_target_ms = std::max(0, value);
        }
    ).set_env("LLAMA_ARG_PREFILL_TBT_TARGET_MS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: sglang/python/sglang/srt/managers/scheduler_components/output_streamer.py:413-445 (a chunk is sent every N tokens, the
        // first token at once, a finished request always); the fork counts UNSENT tokens so a step that adds several (speculative decoding)
        // cannot step over the boundary. Apache-2.0, behaviour only, no code copied. Row S69-ENGREV-SGLANG-STREAM-INTERVAL-2026-10-27-01.
        {"--stream-interval"}, "N",
        string_format("a streaming response sends a chunk only once N generated tokens are unsent (the first token is sent at once, the last chunk is "
            "always sent); a request can set its own with the stream_interval field. Ignored for a request that asks for per-token data "
            "(n_probs, return_tokens) or uses stop_regex (default: %d = every token)", params.stream_interval),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("error: invalid value for --stream-interval (must be >= 1)\n");
            }
            params.stream_interval = value;
        }
    ).set_env("LLAMA_ARG_STREAM_INTERVAL").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: tmp/_tca_ollama_llama_server_0930.go (ollama llm/llama_server.go, MIT, fetched in full) for how a
        // request waits for a free runner slot; the deadline is reimplemented from row S69-Q9-PICK-REQUEST-QUEUE-DEADLINE-2026-09-30-01.
        {"--max-queue-wait-ms"}, "N",
        string_format("a request that is still waiting for a free slot after N milliseconds is answered with HTTP 503 and dropped; "
            "a request can set its own with the max_queue_wait_ms field. A running request is never stopped (default: %d, 0 = wait forever)",
            params.max_queue_wait_ms),
        [](common_params & params, int value) {
            params.max_queue_wait_ms = std::max(0, value);
        }
    ).set_env("LLAMA_ARG_MAX_QUEUE_WAIT_MS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: tf-serving@4438d84 tensorflow_serving/batching/batching_session.cc:746-783 (one deadline per request, counted from
        // enqueue, covering queue wait and execution; Apache-2.0, no code copied). Row S69-TFSERVING-PICK-END-TO-END-DEADLINE-2026-10-15-01.
        {"--request-deadline-ms"}, "N",
        string_format("end-to-end budget of a completion request in milliseconds, counted from its arrival: queue wait + prompt processing + "
            "generation. A request that is still queued or still reading its prompt when the budget runs out is answered with an error; a "
            "request that is generating is stopped and returned with finish_reason \"deadline\". A request can set its own with t_deadline_ms "
            "(default: %d, 0 = none)", params.request_deadline_ms),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value: cannot be negative");
            }
            params.request_deadline_ms = value;
        }
    ).set_env("LLAMA_ARG_REQUEST_DEADLINE_MS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // DISABLE-REVERSIBLE: nothing is switched off here; these new admission flags are opt-in (off by default behind their own
        // LLAMA_ARG_* env), so existing behaviour is unchanged. Rows S69-TFSERVING-PICK-PRIORITY-EVICTION-ON-FULL-QUEUE-2026-10-15-01.
        // SOURCE-READ: tensorflow@master tensorflow/core/kernels/batching_util/shared_batch_scheduler.h:452-503 (PriorityTaskQueue::AddTask;
        // Apache-2.0, no code copied). Row S69-TFSERVING-PICK-PRIORITY-EVICTION-ON-FULL-QUEUE-2026-10-15-01.
        {"--evict-low-priority"},
        {"--no-evict-low-priority"},
        "when --max-queued-requests / --max-queued-tokens would refuse a request, and that request has a strictly higher priority "
        "(lower `priority` value) than a request that is only queued (not running), answer the queued request with HTTP 503 \"evicted\" "
        "and admit the new one instead. The victim is the queued request with the lowest priority, newest first. Equal priority never "
        "evicts (default: not enabled)",
        [](common_params & params, bool value) {
            params.evict_low_priority = value;
        }
    ).set_env("LLAMA_ARG_EVICT_LOW_PRIORITY").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: rtp-llm@95ce33f4 rtp_llm/server/server_args/kv_cache_group_args.py:26-32 (--reserve_block_ratio) and
        // rtp_llm/cpp/cache/CoordinatorCacheManager.cc:74-92,1881-1934 (Apache-2.0, no code copied). Row S69-RTP-LLM-PICK-KV-RESERVE-RATIO-2026-10-15-01.
        {"--kv-reserve-ratio"}, "PERCENT",
        string_format("with --kv-unified: hold back PERCENT of the shared KV pool when a NEW request starts: the request is deferred "
            "while the free cells minus its prompt would drop below that reserve and another request is running (never on an idle "
            "server), so running requests keep room to grow. Complements --kv-admission (default: %d, 0 = not enabled)", params.kv_reserve_ratio),
        [](common_params & params, int value) {
            if (value < 0 || value > 100) {
                throw std::invalid_argument("invalid value: must be 0..100");
            }
            params.kv_reserve_ratio = value;
        }
    ).set_env("LLAMA_ARG_KV_RESERVE_RATIO").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: vllm/v1/core/sched/scheduler.py:741-809 (allocate_slots fails -> preempt a running request) and :1539-1582
        // (_preempt_request: free its blocks, num_computed_tokens = 0, status PREEMPTED, prepend to the waiting queue) @ 924707f1bf94
        // (Apache-2.0, behaviour described, no code copied). Row S69-VLLM-W2-RECOMPUTE-PREEMPTION-INSTEAD-OF-503-2026-10-27-01.
        {"--kv-preempt-requeue"},
        {"--no-kv-preempt-requeue"},
        "with --kv-unified: when the KV pool runs out and a request is preempted, requeue it and recompute it later instead of answering "
        "503, provided it has sent nothing to the client (a non-streaming request; a streaming one already sent its 200 headers and keeps "
        "the 503 / error). Each request is requeued at most --kv-preempt-requeue-max times (default: not enabled)",
        [](common_params & params, bool value) {
            params.kv_preempt_requeue = value;
        }
    ).set_env("LLAMA_ARG_KV_PREEMPT_REQUEUE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--kv-preempt-requeue-max"}, "N",
        string_format("with --kv-preempt-requeue: most times one request is requeued before the next preemption answers 503 (default: %d)",
            params.kv_preempt_requeue_max),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value: must be >= 0");
            }
            params.kv_preempt_requeue_max = value;
        }
    ).set_env("LLAMA_ARG_KV_PREEMPT_REQUEUE_MAX").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: vllm/v1/core/sched/scheduler.py:759-767 (victim = running[-1] under FCFS, max((priority, arrival_time)) under the
        // priority policy) @ 924707f1bf94 (Apache-2.0, behaviour described, no code copied).
        // Row S69-VLLM-W2-PREEMPT-VICTIM-LEAST-PROGRESS-AND-E2E-2026-10-27-01.
        {"--kv-preempt-victim"}, "POLICY",
        "with --kv-unified: which request a KV-exhaustion preemption evicts among requests of the highest priority value: 'largest' = the one "
        "holding most cells (default, frees the most room per eviction), 'least-progress' = the one with the fewest computed cells, "
        "preferring a request that can be requeued over a streaming one (least work thrown away, vLLM-like), 'newest' = the most recently "
        "arrived (vLLM FCFS rule)",
        [](common_params & params, const std::string & value) {
            if (value == "largest")             { params.kv_preempt_victim = 0; }
            else if (value == "least-progress") { params.kv_preempt_victim = 1; }
            else if (value == "newest")         { params.kv_preempt_victim = 2; }
            else { throw std::invalid_argument("invalid value: must be largest, least-progress or newest"); }
        }
    ).set_env("LLAMA_ARG_KV_PREEMPT_VICTIM").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: PaddleNLP csrc/gpu/speculate_decoding_kernels/speculate_step.cu:114-120,125-126 @ 3f87dac9b719 (Apache-2.0, behaviour
        // described, no code copied). Row S69-ENGREV-PADDLENLP-PREEMPT-NEAR-FINISHED-EXEMPT-2026-10-27-01.
        {"--kv-preempt-near-finished"}, "N",
        string_format("with --kv-unified: a request with at most N tokens left to generate is not chosen as the preemption victim while "
            "another request exists (it is about to finish and free its cells; evicting it throws its whole recompute away). If every "
            "request is that close, the --kv-preempt-victim order decides (default: %d, 0 = off)", params.kv_preempt_near_finished),
        [](common_params & params, int value) {
            params.kv_preempt_near_finished = std::max(0, value);
        }
    ).set_env("LLAMA_ARG_KV_PREEMPT_NEAR_FINISHED").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: SGLang b4584ab8 request_metrics_exporter.py, tensorflow/serving server_request_logger.cc (Apache-2.0, no code copied).
        // Row S69-ENGREV-SGLANG-REQUEST-LOG-METRICS-EXPORT-2026-10-27-01.
        {"--request-log"}, "FILE",
        "append one JSON line per finished request (ids, stage times, draft acceptance), keyed by the traceparent trace id or X-Request-Id; "
        "in-flight requests are journalled to FILE.inflight and written to FILE.crash.jsonl after a crash (default: off)",
        [](common_params & params, const std::string & value) {
            params.request_log_path = value;
        }
    ).set_env("LLAMA_ARG_REQUEST_LOG").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--request-log-rate"}, "R",
        string_format("fraction of finished requests written to --request-log, 0..1 (default: %.1f)", params.request_log_rate),
        [](common_params & params, const std::string & value) {
            const double r = std::stod(value);
            if (!(r >= 0.0 && r <= 1.0)) {
                throw std::invalid_argument("--request-log-rate must be within [0, 1]");
            }
            params.request_log_rate = r;
        }
    ).set_env("LLAMA_ARG_REQUEST_LOG_RATE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--request-log-model-rate"}, "NAME=R[,NAME=R...]",
        "per-model override of --request-log-rate (each R within 0..1; a bad entry refuses the option)",
        [](common_params & params, const std::string & value) {
            // validate every NAME=R entry now (same rule as server-request-log.h); the server parses it again at start-up
            size_t pos = 0;
            while (pos <= value.size()) {
                size_t end = value.find(',', pos);
                if (end == std::string::npos) {
                    end = value.size();
                }
                const std::string item = value.substr(pos, end - pos);
                const size_t eq = item.rfind('=');
                if (eq == std::string::npos || eq == 0) {
                    throw std::invalid_argument("--request-log-model-rate: expected NAME=R, got '" + item + "'");
                }
                const double r = std::stod(item.substr(eq + 1));
                if (!(r >= 0.0 && r <= 1.0)) {
                    throw std::invalid_argument("--request-log-model-rate: rate for '" + item.substr(0, eq) + "' must be within [0, 1]");
                }
                pos = end + 1;
            }
            params.request_log_model_rate = value;
        }
    ).set_env("LLAMA_ARG_REQUEST_LOG_MODEL_RATE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: bentoml@517b343 src/bentoml/_internal/marshal/dispatcher.py:75-96,303-311 (Apache-2.0, no code copied).
        // Row S69-BENTOML-PICK-PREDICTIVE-QUEUE-SHEDDING-2026-10-15-01.
        {"--shed-max-latency-ms"}, "N",
        string_format("predictive load shedding: fit service time = a*tokens + b over the last 50 finished requests and, while more than "
            "one request is queued, answer the OLDEST queued request 503 when its wait + the predicted service time of its prompt + "
            "n_predict tokens would reach N milliseconds. Complements --max-queue-wait-ms; waits for 5 measured requests before it "
            "acts (default: %d, 0 = not enabled)", params.shed_max_latency_ms),
        [](common_params & params, int value) {
            params.shed_max_latency_ms = std::max(0, value);
        }
    ).set_env("LLAMA_ARG_SHED_MAX_LATENCY_MS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: mlflow@2d09f29 mlflow/gateway/providers/base.py:713-835 (Apache-2.0, no code copied).
        // Row S69-MLFLOW-PICK-ROUTER-MODE-FAILOVER-CHAIN-2026-10-15-01.
        {"--fallback-model"}, "NAME[,NAME...]",
        "for router server, ordered fallback chain for this model entry (a preset key): when loading the model, or the request before "
        "its first byte is sent, fails, the next model of the chain serves the request. A request that already streamed a chunk is "
        "never retried (default: none)",
        [](common_params & params, const std::string & value) {
            params.fallback_model = value;
        }
    ).set_env("LLAMA_ARG_FALLBACK_MODEL").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--failover-max-attempts"}, "N",
        "for router server, most models tried for one request, the first one included; capped by the chain length (default: 0 = whole chain)",
        [](common_params & params, int value) {
            params.failover_max_attempts = std::max(0, value);
        }
    ).set_env("LLAMA_ARG_FAILOVER_MAX_ATTEMPTS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--control-vector-root"}, "DIR",
        "restrict POST /control-vectors/load to files under DIR (a relative path is taken relative to it); without this any readable path "
        "is accepted (default: unset)",
        [](common_params & params, const std::string & value) {
            params.control_vector_root = value;
        }
    ).set_env("LLAMA_ARG_CONTROL_VECTOR_ROOT").set_examples({LLAMA_EXAMPLE_SERVER}));
    // SOURCE-READ: upstream/koboldcpp/koboldcpp.py:2332-2349 (AGPL, ideas only, no code copied); nuances in server-effort.h
    add_opt(common_arg(
        {"--reasoning-effort-budget"}, "TABLE",
        "turn the OpenAI reasoning_effort level into a thinking-token budget (a fraction of the generation limit), enforced by the "
        "reasoning-budget sampler. TABLE is \"default\" (minimal 10%, low 25%, medium 50%, high 75%) or level=fraction pairs such as "
        "\"low=0.2,high=90%\". A request's own reasoning_budget_tokens wins; \"none\" still turns thinking off; without this flag the "
        "effort level only reaches the chat template (default: unset)",
        [](common_params & params, const std::string & value) {
            params.reasoning_effort_budget = value;
        }
    ).set_env("LLAMA_ARG_REASONING_EFFORT_BUDGET").set_examples({LLAMA_EXAMPLE_SERVER}));
    // row S69-REASONING-ANSWER-RESERVE-OPT-IN-2026-10-17-01; parsed and validated at server load (tools/server/server-answer-reserve.h)
    add_opt(common_arg(
        {"--reasoning-answer-reserve"}, "N|P%",
        "keep room for the answer: when a chat request sets no reasoning budget and no reasoning_effort budget, limit thinking to "
        "the generation limit (max_tokens, capped by --n-predict) minus N tokens, or minus P% of it (e.g. 64 or 25%), floored at 0; "
        "the reasoning-budget sampler then closes the think block (with --reasoning-budget-message). No effect without a finite "
        "limit or on templates without thinking tags. Request field reasoning_answer_reserve overrides it (-1 = off for that "
        "request). Precedence: request budget > effort budget > this > --reasoning-budget (default: -1, off)",
        [](common_params & params, const std::string & value) {
            params.reasoning_answer_reserve = value;
        }
    ).set_env("LLAMA_ARG_REASONING_ANSWER_RESERVE").set_examples({LLAMA_EXAMPLE_SERVER}));
    // Strata serve/server.py @82f46a8c8 (MIT) reasoning_loop_recovery, spec only; see tools/server/server-loop-recovery.h
    add_opt(common_arg(
        {"--reasoning-loop-recovery"}, "MODE",
        "what a loop found while the model is still reasoning does, for requests that do not set reasoning_loop_recovery: "
        "\"off\" (default; repetition_detection works as before), \"stop\" (also run the word-level coverage detector and end the "
        "reply on a loop) or \"recover\" (on the first loop, resume ONCE with the low-effort sentence in place of the high-effort one "
        "in the first system turn, temperature >= 1.0 and presence_penalty >= 1.5; falls back to stop when the sentence is not in "
        "the prompt). The response reports timings.reasoning_recoveries",
        [](common_params & params, const std::string & value) {
            if (value != "off" && value != "stop" && value != "recover") {
                throw std::invalid_argument("--reasoning-loop-recovery must be off, stop or recover");
            }
            params.reasoning_loop_recovery = value;
        }
    ).set_env("LLAMA_ARG_REASONING_LOOP_RECOVERY").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--reasoning-effort-high-text"}, "TEXT",
        "the high-effort sentence the recover retry replaces (default: the Qwen3.8 template's \"Reasoning effort is set to xhigh. ...\")",
        [](common_params & params, const std::string & value) {
            params.reasoning_effort_high_text = value;
        }
    ).set_env("LLAMA_ARG_REASONING_EFFORT_HIGH_TEXT").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--reasoning-effort-low-text"}, "TEXT",
        "the low-effort sentence the recover retry puts in its place (default: the Qwen3.8 template's \"Reasoning effort is set to low. ...\")",
        [](common_params & params, const std::string & value) {
            params.reasoning_effort_low_text = value;
        }
    ).set_env("LLAMA_ARG_REASONING_EFFORT_LOW_TEXT").set_examples({LLAMA_EXAMPLE_SERVER}));
    // SOURCE-READ: upstream/tabbyAPI/common/sampling.py (AGPL, behaviour only, no code copied); nuances in server-overrides.h
    add_opt(common_arg(
        {"--sampler-overrides"}, "FILE",
        "JSON file of operator sampler overrides applied to every completion request: {\"field\": {\"override\": value, \"force\": bool, "
        "\"additive\": bool}}. Without force the value is only a default for requests that omit the field; force replaces what the "
        "request sent (aliases included); additive puts a list value in front of the request's list. Unknown field names are "
        "skipped with a warning at load. Can be replaced at run time with POST /sampler-overrides (admin key) (default: unset)",
        [](common_params & params, const std::string & value) {
            params.sampler_overrides_file = value;
        }
    ).set_env("LLAMA_ARG_SAMPLER_OVERRIDES").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--sampler-overrides-model"}, "FILE",
        "same format as --sampler-overrides, but a per-model layer: its keys replace the same keys of the global file. Meant for a "
        "router preset section, so the layer follows the model it is set for (default: unset)",
        [](common_params & params, const std::string & value) {
            params.sampler_overrides_model_file = value;
        }
    ).set_env("LLAMA_ARG_SAMPLER_OVERRIDES_MODEL").set_examples({LLAMA_EXAMPLE_SERVER}));
    // SOURCE-READ: the vision.py helper under backends/ in the local tabbyAPI clone (AGPL, behaviour only, no code copied); nuances in tools/mtmd/mtmd-embd-cache.h
    add_opt(common_arg(
        {"--mmproj-embd-cache"}, "MiB",
        "keep up to MiB of vision-encoder outputs so an image that returns (same pixels, any prefix, any slot) skips the vision tower. Keyed by a "
        "content hash of the preprocessed image, evicted least-recently-used by bytes; the images of the request being served are never evicted "
        "for the same request's later images (default: 0 = off)",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            params.mmproj_embd_cache_mib = value;
        }
    ).set_env("LLAMA_ARG_MMPROJ_EMBD_CACHE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--embd-length-order"},
        {"--no-embd-length-order"},
        string_format("post the inputs of one /v1/embeddings request longest first so short ones fill the gaps of the micro-batches (results keep request order) (default: %s)",
                      params.embd_length_order ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.embd_length_order = value;
        }
    ).set_env("LLAMA_ARG_EMBD_LENGTH_ORDER").set_examples({LLAMA_EXAMPLE_SERVER}));
    // SOURCE-READ: mlc-ai/mlc-llm cpp/serve/engine_actions/batch_jump_forward.cc (Apache-2.0), idea only; algorithm in tools/server/server-jump-forward.h
    add_opt(common_arg(
        {"--grammar-jump-forward"}, "N",
        "when a user grammar (GBNF) forces the next text, append up to N tokens spelling it and decode them in one batch instead of one decode "
        "per token. The text is exactly the grammar's; the tokens are the forced text tokenised on its own, so they can differ from a plain run's "
        "split. Skipped for requests with logprobs, banned strings, classifier-free guidance or speculative decoding. A request can override it "
        "with \"grammar_jump_forward\" (default: 0 = off)",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            params.grammar_jump_forward = value;
        }
    ).set_env("LLAMA_ARG_GRAMMAR_JUMP_FORWARD").set_examples({LLAMA_EXAMPLE_SERVER}));
    // SOURCE-READ: sgl-project/sglang python/sglang/srt/constrained/json_schema_validation.py:94-260 (Apache-2.0), idea only;
    // the keyword list is this fork's own converter's, see common/json-schema-strict.h
    add_opt(common_arg(
        {"--json-schema-strict"},
        "refuse (HTTP 400) a json_schema whose constraints the grammar cannot enforce (multipleOf, uniqueItems, not, unknown "
        "string formats, number bounds, ...), naming each by its JSON pointer, instead of silently dropping them. A request can "
        "override it with \"json_schema_strict\" (default: off = accept and log)",
        [](common_params & params) {
            params.json_schema_strict = true;
        }
    ).set_env("LLAMA_ARG_JSON_SCHEMA_STRICT").set_examples({LLAMA_EXAMPLE_SERVER}));
    // SOURCE-READ: alibaba/MNN omni.cpp (Apache-2.0) video_max_frames / video_max_vision_tokens, idea only; planner and nuances in tools/mtmd/mtmd-video-plan.h
    add_opt(common_arg(
        {"--video-max-frames"}, "N",
        "a video input is read as at most N frames: the sampling rate is lowered to fit, and the count is kept a multiple of the model's temporal "
        "group (default: 0 = no cap)",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            params.video_max_frames = value;
        }
    ).set_env("LLAMA_ARG_VIDEO_MAX_FRAMES").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_MTMD}));
    add_opt(common_arg(
        {"--video-max-tokens"}, "N",
        "a video input must fit N vision tokens in the LLM (items x tokens per item): frames are scaled down by whole encoder cells before they are "
        "read, and a budget too small for one item is rejected up front with the numbers (default: 0 = no budget)",
        [](common_params & params, const std::string & value) {
            const long long n = std::stoll(value);
            if (n < 0) {
                throw std::invalid_argument("invalid value");
            }
            params.video_max_tokens = (int64_t) n;
        }
    ).set_env("LLAMA_ARG_VIDEO_MAX_TOKENS").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_MTMD}));
    // SOURCE-READ: upstream/LocalAI/core/config/inference_defaults.go (MIT); matching rules and nuances in sampling-family-defaults.h
    add_opt(common_arg(
        {"--sampling-family-defaults"},
        "when the model file carries no general.sampling.* values, use the vendor-recommended temperature / top-p / top-k / min-p / repeat and presence "
        "penalty of the model family recognised from its name (alias, general.name, file name, repo). A sampling flag you pass always wins, the "
        "model's own general.sampling.* values win over the table, and every value taken is logged. Off by default: choosing sampling from a file "
        "name changes behaviour, so reproducible runs and benchmarks must not rely on it (default: off)",
        [](common_params & params) {
            params.sampling_family_defaults = true;
        }
    ).set_env("LLAMA_ARG_SAMPLING_FAMILY_DEFAULTS").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    // SOURCE-READ: upstream/aphrodite-engine/aphrodite/v1/spec_decode/dynamic/utils.py (AGPL, behaviour only, no code copied); nuances in server-draft-schedule.h
    add_opt(common_arg(
        {"--draft-max-by-slots"}, "SPEC",
        "speculative draft length by the number of slots generating together, e.g. \"1:5,3:2,9:0\" (1-2 slots draft up to 5 tokens, 3-8 slots up to 2, "
        "9 or more none) or with explicit ranges \"1-4:5,8-16:2\" (a gap carries the previous value, the tail the last). A long draft pays when the "
        "device has slack and costs throughput when many slots verify at once. The schedule only lowers a slot's own budget: --draft-max, the "
        "request's speculative.n_max and the context room still apply. Entries ascending, no overlap, the first starting at 1 (default: unset)",
        [](common_params & params, const std::string & value) {
            params.draft_max_by_slots = value;
        }
    ).set_env("LLAMA_ARG_DRAFT_MAX_BY_SLOTS").set_examples({LLAMA_EXAMPLE_SERVER}));
    // idea: RunanywhereAI/trnrt@ef581f5 config.py:130-135 spec_hysteresis (Apache-2.0), own implementation in server-draft-schedule.h
    add_opt(common_arg(
        {"--draft-max-by-slots-hysteresis"}, "N",
        "with --draft-max-by-slots: when the generating-slot count falls back across a range boundary, raise the draft length again only once "
        "the count is N slots below that boundary; lowering stays immediate. Stops the draft length flip-flopping when the load oscillates around "
        "a boundary (default: 0 = off)",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("--draft-max-by-slots-hysteresis must be >= 0");
            }
            params.draft_max_by_slots_hysteresis = value;
        }
    ).set_env("LLAMA_ARG_DRAFT_MAX_BY_SLOTS_HYSTERESIS").set_examples({LLAMA_EXAMPLE_SERVER}));
    // DISABLE-REVERSIBLE: --disable-management-routes is itself the named switch (default off); RESEARCH-DONE: pytorch/serve README
    // (management API off by default), row S69-Q9-PICK-MANAGEMENT-PLANE-DEFAULT-OFF-CONTROL-VECTOR-PATH-2026-09-30-01
    add_opt(common_arg(
        {"--disable-management-routes"},
        "answer 403 on the routes that change the server or its files: model load/unload (router), control vectors, LoRA hotswap, "
        "slot save/restore/erase, saved prompts, the SQL session store and POST /props. Read-only routes keep working. "
        "Recommended on a shared host, where an API key (or, with no key, anyone who can reach the port) would otherwise be able to "
        "make the server read or write a file (default: off)",
        [](common_params & params) {
            params.server_disable_management_routes = true;
        }
    ).set_env("LLAMA_ARG_DISABLE_MANAGEMENT_ROUTES").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // "-ctx-ckpt-e/--ctx-checkpoints-eviction": the fork's spelling (pre-merge common.cpp @
        // 35f0cac75, field ctx_checkpoint_eviction -- old server-context.cpp:3853-3854,3791-3834
        // @ 35f0cac75, evict_checkpoint_by_variance()). No upstream equivalent: tools/server's own
        // create_checkpoint() always evicted the oldest checkpoint (FIFO); "variance"/"auto" instead
        // pick the checkpoint whose removal leaves the most even spacing.
        {"-ctx-ckpt-e", "--ctx-checkpoints-eviction"}, "{auto,fifo,variance}",
        "policy for which context checkpoint to evict when the per-slot limit is reached: "
        "fifo = oldest first, variance = evict the checkpoint that leaves spacing most even "
        "(default: variance, same as auto)",
        [](common_params & params, const std::string & value) {
            params.ctx_checkpoint_eviction = common_checkpoint_eviction_from_name(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // Disk spill of context checkpoints. SOURCE-READ: ik_llama.cpp main (PR 2446, MIT) examples/server/server-context.cpp
        // :36-60, :464-476, :2825-2874, :3849-3866, :3929-3972 and common/common.cpp :3081-3090 (fetched 2026-10-05).
        // RESEARCH-DONE: https://raw.githubusercontent.com/ikawrakow/ik_llama.cpp/main/examples/server/server-context.cpp
        // Reimplemented for the split server's common_prompt_checkpoint (three data vectors), own file format.
        // Row S69-FLAG-WIRING-CTX-CKPT-SPILL-NOT-PORTED-2026-10-01-01. This ADDS a feature; nothing is switched off.
        {"--ctx-ckpt-spill-dir", "--ctx-checkpoints-spill-dir"}, "DIR",
        "spill evicted context checkpoints to files in DIR (e.g. on NVMe) instead of dropping them, so they can be reloaded "
        "instead of re-processing the prompt; the directory is created, and stale ckpt-s* files in it are removed at startup "
        "(default: empty = no spill)",
        [](common_params & params, const std::string & value) {
            params.ctx_checkpoint_spill_dir = value;
        }
    ).set_env("LLAMA_ARG_CTX_CKPT_SPILL_DIR").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--ctx-ckpt-cram-soft"},
        "keep cold context checkpoints (beyond --ctx-ckpt-live-n per slot) in a compressed read-only block store in RAM (the software "
        "CRAM tier, src/llama-cram-soft.cpp) instead of raw; reads are checksummed and bit-identical; ignored when --ctx-ckpt-spill-dir is set "
        "(default: off)",
        [](common_params & params) {
            params.ctx_checkpoint_cram_soft = true;
        }
    ).set_env("LLAMA_ARG_CTX_CKPT_CRAM_SOFT").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--ctx-ckpt-dominance"},
        "when a request releases its slot, free the context checkpoints that a deeper one on the same unbranched path dominates "
        "(an older checkpoint then serves only a request that branches there); keeps the newest prompt-end checkpoint and every "
        "branch-point checkpoint and the ones below it. Rule from trnrt radix_cache.free_dominated (Apache-2.0), reimplemented "
        "(default: off)",
        [](common_params & params) {
            params.ctx_checkpoint_dominance = true;
        }
    ).set_env("LLAMA_ARG_CTX_CKPT_DOMINANCE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cram-probe"},
        "at model load, log what the read-only sysfs probe finds about private / compressed memory tiers (Meta CRAM). The sysfs names it "
        "checks are GUESSES (the kernel series is not merged), so a negative result proves nothing about the hardware. Diagnostic only; default: off",
        [](common_params & params) {
            params.cram_probe = true;
        }
    ).set_env("LLAMA_ARG_CRAM_PROBE"));
    add_opt(common_arg(
        {"--vram-spill"}, "MODE",
        "where device allocations >= --vram-spill-threshold go when VRAM is short: off | managed (CUDA managed/unified memory) | "
        "hostmap (pinned host RAM read by the GPU over PCIe, zero-copy) | auto (hostmap only when the allocation does not fit in free VRAM minus "
        "--vram-spill-reserve). Also adds the buffer type CUDA<n>_Spill for -ot. CUDA builds only (UNTESTED on hardware); on other builds it is "
        "recorded and logged but has no effect. WDDM/Windows managed memory cannot oversubscribe VRAM, prefer hostmap there. default: off",
        [](common_params & params, const std::string & value) {
            ggml_vram_spill_mode m;
            if (!ggml_vram_spill_parse_mode(value.c_str(), &m)) {
                throw std::invalid_argument("invalid --vram-spill mode '" + value + "' (expected off|managed|hostmap|auto)");
            }
            params.vram_spill_mode = value;
        }
    ).set_env("LLAMA_ARG_VRAM_SPILL"));
    add_opt(common_arg(
        {"--vram-spill-threshold"}, "MiB",
        string_format("with --vram-spill: allocations smaller than this stay in VRAM (default: %" PRId64 ")", params.vram_spill_threshold_mib),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            params.vram_spill_threshold_mib = value;
        }
    ).set_env("LLAMA_ARG_VRAM_SPILL_THRESHOLD"));
    add_opt(common_arg(
        {"--vram-spill-reserve"}, "MiB",
        string_format("with --vram-spill auto: VRAM kept free for compute buffers before anything spills (default: %" PRId64 ")", params.vram_spill_reserve_mib),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            params.vram_spill_reserve_mib = value;
        }
    ).set_env("LLAMA_ARG_VRAM_SPILL_RESERVE"));
    add_opt(common_arg(
        {"--ctx-ckpt-live-n", "--ctx-checkpoints-live-n"}, "N",
        string_format("with --ctx-ckpt-spill-dir: the most checkpoints per slot kept in memory, the rest are spilled "
            "(default: %d, <0 = keep all in memory)", params.ctx_checkpoint_ram_live),
        [](common_params & params, int value) {
            params.ctx_checkpoint_ram_live = value;
        }
    ).set_env("LLAMA_ARG_CTX_CKPT_LIVE_N").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-cram", "--cache-ram"}, "N",
        string_format("set the maximum cache size in MiB (default: %d, 0 for decision models, -1 - no limit, 0 - disable)"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/16391)", params.cache_ram_mib),
        [](common_params & params, int value) {
            params.cache_ram_mib = value;
            params.cache_ram_explicit = true;
        }
    ).set_env("LLAMA_ARG_CACHE_RAM").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        // SOURCE-READ: engines/vllm/vllm/v1/kv_offload/cpu/policies/arc.py (recent / frequent split) and
        // engines/sglang/python/sglang/srt/mem_cache/utils.py:48-84 (SLRU among the offered policies). Apache-2.0, behaviour only.
        // Row S69-VLLM-PICK-PROMPT-CACHE-SLRU-EVICTION-2026-09-29-01.
        {"--cache-ram-slru"},
        "scan-resistant eviction for the host prompt cache: entries restored at least once are kept over entries never restored, so a burst "
        "of one-shot long prompts cannot push out a shared system prompt (default: off = plain LRU)",
        [](common_params & params) {
            params.cache_ram_slru = true;
        }
    ).set_env("LLAMA_ARG_CACHE_RAM_SLRU").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: engines/rtp-llm/docs/backend/reuse_kv_cache.md (MultiTaskPrompt: prompts given at startup are prefilled once and
        // kept resident; startup ERRORS instead of a silent downgrade when a limit is not met). Apache-2.0, behaviour only.
        // Row S69-rtp-llm-PICK-STARTUP-WARMED-PINNED-PREFIX-CACHE-2026-09-29-01.
        {"--warm-prefix-file"}, "FNAME",
        "JSON list of {\"id\", \"messages\" or \"prompt\"}: each is run through the normal request path once after the model loads and its "
        "prompt state is pinned in the host prompt cache (never evicted). Needs --cache-ram > 0; the server stops with an error when the "
        "pinned bytes would exceed --warm-prefix-share of --cache-ram (default: empty = off)",
        [](common_params & params, const std::string & value) {
            params.warm_prefix_file = value;
        }
    ).set_env("LLAMA_ARG_WARM_PREFIX_FILE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--warm-prefix-share"}, "F",
        string_format("with --warm-prefix-file: the largest fraction of --cache-ram the pinned prefixes may hold (default: %.2f, range 0.05-0.95)",
                      (double) params.warm_prefix_share),
        [](common_params & params, const std::string & value) {
            const float f = std::stof(value);
            if (!(f >= 0.05f && f <= 0.95f)) {
                throw std::invalid_argument("--warm-prefix-share must be between 0.05 and 0.95");
            }
            params.warm_prefix_share = f;
        }
    ).set_env("LLAMA_ARG_WARM_PREFIX_SHARE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // NVMe tier under the host prompt cache. Row S69-TIERED-KV-DISK-TIER-SLICE1-2026-10-05-01; design in common/disk-tier.h.
        // SOURCE-READ: tmp/research/tiered1005/oss_sglang_hicache_design.txt (docs.sglang.io hicache_design.html,
        // Apache-2.0, read in full) and oss_dynamo_kvbm_design.txt (NVIDIA Dynamo KVBM design, Apache-2.0, read in full). No code copied.
        // SOURCE-NUANCE: HiCache's write_through_selective backs a block up only after a hit-count threshold, and its prefetch
        // has a configurable minimum hit length (default 256 tokens) -- the reason --cache-disk-min-tokens exists; its default here is
        // 32, from a measurement on this host (MEASURED: temp/_c2_1005_rb_*.txt crossover 3-20 tokens), not HiCache's 256. The
        // description "a disk tier" says neither.
        {"--cache-disk-dir"}, "DIR",
        "spill prompt states that --cache-ram must evict to checksummed files in DIR (e.g. on NVMe) instead of dropping them, "
        "and restore them when a later request shares their prefix; the directory is created, stale pcache-* files in it are removed "
        "at startup and every file is removed at exit (default: empty = off)",
        [](common_params & params, const std::string & value) {
            params.cache_disk_dir = value;
        }
    ).set_env("LLAMA_ARG_CACHE_DISK_DIR").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-disk-size"}, "MiB",
        string_format("with --cache-disk-dir: the most bytes the directory may hold (default: %d)", params.cache_disk_mib),
        [](common_params & params, int value) {
            params.cache_disk_mib = value;
        }
    ).set_env("LLAMA_ARG_CACHE_DISK_SIZE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-disk-min-free"}, "MiB",
        "with --cache-disk-dir: never let the tier take its volume below this much free space; a spill that would is "
        "refused and counted in llamacpp:prompt_cache_disk_refused_space_total. -1 = built-in floor (10 GiB), "
        "0 = no floor (default: -1)",
        [](common_params & params, int value) {
            if (value < -1) {
                throw std::invalid_argument("--cache-disk-min-free must be -1, 0 or a size in MiB");
            }
            params.cache_disk_min_free_mib = value;
        }
    ).set_env("LLAMA_ARG_CACHE_DISK_MIN_FREE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--beam-mode"}, "vllm|ct2",
        string_format("server beam search (request field use_beam_search): ranking and stopping rules. vllm = run every beam to "
            "n_predict (vLLM's loop); ct2 = CTranslate2's loop: EOS in the top width candidates finishes a hypothesis and is replaced "
            "from the spare candidates, patience-scaled early stop, min_tokens, soft prefix bias (default: %s)", params.beam_mode.c_str()),
        [](common_params & params, const std::string & value) {
            if (value != "vllm" && value != "ct2") {
                throw std::invalid_argument("--beam-mode must be vllm or ct2");
            }
            params.beam_mode = value;
        }
    ).set_env("LLAMA_ARG_BEAM_MODE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-disk-min-tokens"}, "N",
        string_format("with --cache-disk-dir: never spill a prompt shorter than N tokens, re-prefilling it costs less than the "
            "disk round trip; measured crossover on CPU hosts is 3-20 tokens (default: %d)", params.cache_disk_min_tokens),
        [](common_params & params, int value) {
            params.cache_disk_min_tokens = value;
        }
    ).set_env("LLAMA_ARG_CACHE_DISK_MIN_TOKENS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // restore-versus-recompute rule, row S69-TIERED-RESTORE-VS-RECOMPUTE-POLICY-2026-10-05-01 (LMCache paper s8.7 and Mooncake
        // Algorithm 1 compare transfer time with prefill time; read in full by the previous agent, captures in temp/research/tiered1005).
        // MEASURED: restore 6-190 ms against recompute 50-8600 ms on 4 real models, temp/_c2_1005_rb_*.txt.
        {"--cache-disk-min-speedup"}, "X",
        string_format("with --cache-disk-dir: spill or restore a state only while reading it back takes at most 1/X of the time to "
            "recompute it, using the restore speed and prompt-processing speed measured on this server; 0 turns the rule off, "
            "the other limits still apply (default: %.1f)", (double) params.cache_disk_min_speedup),
        [](common_params & params, const std::string & value) {
            params.cache_disk_min_speedup = std::stof(value);
        }
    ).set_env("LLAMA_ARG_CACHE_DISK_MIN_SPEEDUP").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-disk-write-budget"}, "MiB_PER_HOUR",
        "with --cache-disk-dir: SSD wear budget. Spilling stops while more than this many MiB per hour would be written "
        "(a token bucket that holds one hour; the bytes written are exported as llamacpp:prompt_cache_disk_written_bytes_total) "
        "(default: 0 = unlimited)",
        [](common_params & params, int value) {
            params.cache_disk_write_mib_per_hour = value;
        }
    ).set_env("LLAMA_ARG_CACHE_DISK_WRITE_BUDGET").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // SOURCE-READ: DeepSpeed csrc/aio/common/deepspeed_aio_common.cpp:192-233 (do_aio_operation_overlap, Apache-2.0): keep I/O in flight
        // while other work proceeds. Applied here to the disk tier's checksum stage. Row S69-LINK-STORAGE-OFFLOAD-1014. No code copied.
        {"--cache-disk-io-chunk"}, "MiB",
        "with --cache-disk-dir: read and write spilled states in chunks of this many MiB and checksum each chunk on this thread while a "
        "helper thread moves the next one, instead of moving the whole state and then checksumming it. States no larger than one chunk "
        "are unaffected; the files are byte-identical (default: 0 = serial)",
        [](common_params & params, int value) {
            params.cache_disk_io_chunk_mib = value;
        }
    ).set_env("LLAMA_ARG_CACHE_DISK_IO_CHUNK").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // ik_llama.cpp #2538 (MIT): the option and its meaning; our parser spelling.
        {"-dsv4ls", "--dsv4-legacy-state"},
        string_format("write DeepSeek-V4 state in the legacy full-slice layout (no magic, ratios or row counts), byte-identical to the "
            "older format and readable by older builds (default: %s)", params.dsv4_legacy_state ? "true" : "false"),
        [](common_params & params) {
            params.dsv4_legacy_state = true;
        }
    ).set_env("LLAMA_ARG_DSV4_LEGACY_STATE").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-kvu", "--kv-unified"},
        {"-no-kvu", "--no-kv-unified"},
        "use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)",
        [](common_params & params, bool value) {
            params.kv_unified = value;
        }
    ).set_env("LLAMA_ARG_KV_UNIFIED").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_BATCHED, LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL, LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--kv-admission"},
        {"--no-kv-admission"},
        "with --kv-unified: before starting a request, project the shared pool's PEAK use over the remaining "
        "generation of every in-flight request plus the new one (LightLLM's rule), and defer the new request "
        "while that peak exceeds the pool. Prevents the exhaustion that otherwise preempts a request mid-flight. "
        "A request without n_predict is projected to its slot's full context, so unbounded requests serialise "
        "(default: disabled)",
        [](common_params & params, bool value) {
            params.kv_admission = value;
        }
    ).set_env("LLAMA_ARG_KV_ADMISSION").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--kv-admission-decode-tokens"}, "N",
        string_format("with --kv-admission: project at most N decode tokens per request (in flight or new) instead of its "
            "whole n_predict / slot context, so requests that set no limit, or a large one they rarely reach, are admitted "
            "side by side instead of one at a time. A request that does run past N can still exhaust the pool; the "
            "--kv-unified preemption then handles it as it does without admission (default: %d, 0 = no cap)",
            params.kv_admission_decode_tokens),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("--kv-admission-decode-tokens must be >= 0");
            }
            params.kv_admission_decode_tokens = value;
        }
    ).set_env("LLAMA_ARG_KV_ADMISSION_DECODE_TOKENS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-idle-slots"},
        {"--no-cache-idle-slots"},
        "save idle slots to the prompt cache on new task, and clear them when using unified KV (default: enabled, requires cache-ram)",
        [](common_params & params, bool value) {
            params.cache_idle_slots = value;
        }
    ).set_env("LLAMA_ARG_CACHE_IDLE_SLOTS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-publish-on-prefill"},
        {"--no-cache-publish-on-prefill"},
        "proactively publish a slot's prompt cache the instant its prefill finishes, so a "
        "concurrently-active sibling can reuse the shared prefix while this slot is still "
        "generating (SGLang RadixAttention gap port); always skipped when there is only one slot "
        "(default: enabled, requires cache-ram and more than one slot)",
        [](common_params & params, bool value) {
            params.cache_publish_on_prefill = value;
        }
    ).set_env("LLAMA_ARG_CACHE_PUBLISH_ON_PREFILL").set_examples({LLAMA_EXAMPLE_SERVER}));
    // DISABLE-REVERSIBLE: --radix-cache (default disabled) is the new feature's own opt-in flag,
    // same pattern as --cfg-guidance-slots above; the model-type/SWA auto-disable below is a
    // runtime SAFETY gate (row's CHALLENGE section: shared position range != shared meaning for
    // recurrent/hybrid/diffusion/SWA memory), re-enable condition is simply not using those.
    add_opt(common_arg(
        {"--radix-cache"},
        {"--no-radix-cache"},
        "on-device token-level radix prefix cache: two DIFFERENT prompts sharing only a prefix "
        "(e.g. a common system prompt) both reuse the shared KV via zero-copy seq_cp, not just an "
        "identical resend (contrast --cache-publish-on-prefill/server_prompt_cache, which "
        "serializes whole states through host RAM). Requires --kv-unified; refused otherwise, and "
        "auto-disabled with a warning for recurrent/hybrid/diffusion models or SWA (n_swa > 0), "
        "where a shared position range does not mean shared meaning "
        "(default: disabled)",
        [](common_params & params, bool value) {
            params.radix_cache = value;
        }
    ).set_env("LLAMA_ARG_RADIX_CACHE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--radix-cache-slots"}, "N",
        string_format("number of extra sequence ids reserved for the radix cache's own tree nodes, "
                       "on top of --parallel and --cfg-guidance-slots (default: %d). Ignored unless "
                       "--radix-cache is set. Each node in the tree pins its token range's KV cells "
                       "under one reserved id until evicted (LRU over idle leaves), so this bounds "
                       "how many distinct branch points the tree can hold at once, not how many "
                       "tokens -- a single id can own an arbitrarily long unbranched chain",
                       params.n_radix_cache_slots),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("error: invalid value for --radix-cache-slots (must be >= 0)\n");
            }
            params.n_radix_cache_slots = value;
        }
    ).set_env("LLAMA_ARG_RADIX_CACHE_SLOTS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--lora-pool-dir"}, "DIR",
        "server: directory of LoRA adapters served on demand. A request that carries \"lora_adapter\":\"NAME\" (and optionally \"lora_adapter_scale\") runs with DIR/NAME.gguf, "
        "loaded when first asked for and evicted least-recently-used when the pool is full (see --lora-pool-max, --lora-pool-max-mib). A request whose adapter cannot be made "
        "resident because every resident adapter is in use gets HTTP 503. NAME may contain only letters, digits, '.', '_' and '-'. (default: off)",
        [](common_params & params, const std::string & value) {
            params.lora_pool_dir = value;
        }
    ).set_env("LLAMA_ARG_LORA_POOL_DIR").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--lora-pool-max"}, "N",
        string_format("server: most adapters resident in the --lora-pool-dir pool at once, 0 = no limit (default: %d)", params.lora_pool_max),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("error: invalid value for --lora-pool-max (must be >= 0)\n");
            }
            params.lora_pool_max = value;
        }
    ).set_env("LLAMA_ARG_LORA_POOL_MAX").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--lora-pool-max-mib"}, "N",
        "server: byte budget of the --lora-pool-dir pool in MiB of adapter files, 0 = no limit (default: 0)",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("error: invalid value for --lora-pool-max-mib (must be >= 0)\n");
            }
            params.lora_pool_max_mib = value;
        }
    ).set_env("LLAMA_ARG_LORA_POOL_MAX_MIB").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--lora-hetero-batch"},
        "server: let slots whose requests use DIFFERENT LoRA adapters share one decode batch (per-sequence adapters: every adapter in use is evaluated on the whole batch "
        "and masked per token, so it pays off for a few adapters at a time). Not supported with speculative decoding, aLoRA or the llama.cpp flash-attention path; "
        "(default: off, slots with different adapters run in separate batches)",
        [](common_params & params) {
            params.lora_hetero_batch = true;
        }
    ).set_env("LLAMA_ARG_LORA_HETERO_BATCH").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--token-ban-substr"},
        "server: honour the request field \"banned_tokens\" (list of strings, koboldcpp behaviour): a short string that is one token bans EVERY vocabulary entry that contains it "
        "(case-insensitive substring); any other string is banned as a phrase through the banned_strings machinery. (default: off, the field is ignored)",
        [](common_params & params) {
            params.token_ban_substr = true;
        }
    ).set_env("LLAMA_ARG_TOKEN_BAN_SUBSTR").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--prompt-reuse-plan"},
        "server: decide how much of a slot's cached prompt is reused with the koboldcpp-style planner (keeps at least one prompt token to evaluate, applies --prompt-reuse-min and "
        "--prompt-reuse-keep-input, and for recurrent models reuses only a cache that is a complete prefix of the prompt). (default: off, plain longest-common-prefix reuse)",
        [](common_params & params) {
            params.prompt_reuse_plan = true;
        }
    ).set_env("LLAMA_ARG_PROMPT_REUSE_PLAN").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--prompt-reuse-min"}, "N",
        "server, with --prompt-reuse-plan: reuse of fewer than N cached prompt tokens is discarded (default: 0)",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("error: invalid value for --prompt-reuse-min (must be >= 0)\n");
            }
            params.prompt_reuse_min = value;
        }
    ).set_env("LLAMA_ARG_PROMPT_REUSE_MIN").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--prompt-reuse-keep-input"}, "N",
        "server, with --prompt-reuse-plan: always leave at least N prompt tokens to be evaluated, whatever the cache holds (default: 0)",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("error: invalid value for --prompt-reuse-keep-input (must be >= 0)\n");
            }
            params.prompt_reuse_keep_input = value;
        }
    ).set_env("LLAMA_ARG_PROMPT_REUSE_KEEP_INPUT").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--asr-quality"},
        "server: gate every /v1/audio/transcriptions result (compression ratio of the text, mean token logprob, tail entropy) and, when it fails, decode the audio again at the "
        "next temperature of the whisper ladder (--asr-temperature-inc); the best attempt is returned. (default: off)",
        [](common_params & params) {
            params.asr_quality = true;
        }
    ).set_env("LLAMA_ARG_ASR_QUALITY").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--asr-temperature-inc"}, "F",
        string_format("server, with --asr-quality: temperature added per retry (default: %.1f)", (double) params.asr_temperature_inc),
        [](common_params & params, const std::string & value) {
            const float v = std::stof(value);
            if (!(v > 0.0f)) {
                throw std::invalid_argument("error: invalid value for --asr-temperature-inc (must be > 0)\n");
            }
            params.asr_temperature_inc = v;
        }
    ).set_env("LLAMA_ARG_ASR_TEMPERATURE_INC").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--asr-compression-ratio-thold"}, "F",
        string_format("server, with --asr-quality: a transcript whose zlib compression ratio exceeds F is treated as looping and retried (default: %.1f)", (double) params.asr_compression_ratio_thold),
        [](common_params & params, const std::string & value) {
            params.asr_compression_ratio_thold = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ASR_COMPRESSION_RATIO_THOLD").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--asr-logprob-thold"}, "F",
        string_format("server, with --asr-quality: a transcript whose mean token logprob is below F is retried (default: %.1f)", (double) params.asr_logprob_thold),
        [](common_params & params, const std::string & value) {
            params.asr_logprob_thold = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ASR_LOGPROB_THOLD").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--asr-entropy-thold"}, "F",
        string_format("server, with --asr-quality: a transcript of more than 32 tokens whose last 32 tokens have an entropy below F bits is retried, 0 = gate off (default: %.1f)", (double) params.asr_entropy_thold),
        [](common_params & params, const std::string & value) {
            params.asr_entropy_thold = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ASR_ENTROPY_THOLD").set_examples({LLAMA_EXAMPLE_SERVER}));
    // Named after whisper.cpp server's --vad-model / --vad-threshold (examples/server/server.cpp:188-189, 261-262), with the asr-
    // prefix of the other transcription switches; one flag both names the model and turns the gate on.
    add_opt(common_arg(
        {"--asr-vad"}, "FNAME",
        "server: voice-activity gate on /v1/audio/transcriptions with the native Silero VAD (whisper.cpp ggml-format file, e.g. "
        "ggml-silero-v6.2.0.bin). An upload with no speech segment returns an empty transcript without running the model; "
        "speech is transcribed as before. Loaded once at start; the server refuses to start if it does not load. (default: off)",
        [](common_params & params, const std::string & value) {
            params.asr_vad_model = value;
        }
    ).set_env("LLAMA_ARG_ASR_VAD").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--asr-vad-threshold"}, "F",
        string_format("server, with --asr-vad: Silero speech probability at which speech starts, in (0, 1) (default: %.2f)", (double) params.asr_vad_threshold),
        [](common_params & params, const std::string & value) {
            const float v = std::stof(value);
            if (!(v > 0.0f && v < 1.0f)) {
                throw std::invalid_argument("error: invalid value for --asr-vad-threshold (must be in (0, 1))\n");
            }
            params.asr_vad_threshold = v;
        }
    ).set_env("LLAMA_ARG_ASR_VAD_THRESHOLD").set_examples({LLAMA_EXAMPLE_SERVER}));
    // whisper.cpp trims by default once VAD is on (src/whisper.cpp, VAD path of whisper_full); here it is a separate opt-in
    // because the gate alone (--asr-vad) is already in use and trimming changes what the model hears.
    add_opt(common_arg(
        {"--asr-vad-trim"},
        "server, with --asr-vad: transcribe just the speech spans the VAD found (silence is cut out before the model runs) "
        "and map the segment times of srt / vtt / verbose_json back to the upload. (default: off, the whole upload is transcribed)",
        [](common_params & params) {
            params.asr_vad_trim = true;
        }
    ).set_env("LLAMA_ARG_ASR_VAD_TRIM").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--asr-vad-trim-pad-ms"}, "N",
        string_format("server, with --asr-vad-trim: milliseconds of silence written between two kept speech spans, 0 = joined with no gap; "
                      "a pause no longer than N is kept as it is (default: %d)", (int) params.asr_vad_trim_pad_ms),
        [](common_params & params, int value) {
            if (value < 0 || value > 5000) {
                throw std::invalid_argument("error: invalid value for --asr-vad-trim-pad-ms (must be in [0, 5000])\n");
            }
            params.asr_vad_trim_pad_ms = value;
        }
    ).set_env("LLAMA_ARG_ASR_VAD_TRIM_PAD_MS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--asr-native-prompt"}, "MODE",
        "server: /v1/audio/transcriptions prompt form. 'on' uses the model's own transcription prompt, "
        "[INST]<audio>[/INST]lang:<language>[TRANSCRIBE] (Voxtral; mistral-common's transcription request), when the vocabulary has a "
        "[TRANSCRIBE] control token (bare transcript, no chat wrapping), and warns and keeps the chat template otherwise; 'off' uses "
        "the chat template with a transcribe instruction. 'on' does not detect silence: on a silent clip the model writes invented "
        "text. A request that sends its own `prompt` always goes through the chat template. 'vibevoice' renders VibeVoice-ASR's "
        "offline prompt (system line, audio block, \"This is a N.NN seconds audio, please transcribe it with these keys: Start time, "
        "End time, Speaker ID, Content\", no assistant header), turns the request `prompt` into its \"with extra info\" hotword "
        "clause, and RMS-normalises the upload to -25 dBFS first (default: off)",
        [](common_params & params, const std::string & value) {
            if (value != "on" && value != "off" && value != "vibevoice") {
                throw std::invalid_argument("error: --asr-native-prompt must be on, off or vibevoice\n");
            }
            params.asr_native_prompt = value;
        }
    ).set_env("LLAMA_ARG_ASR_NATIVE_PROMPT").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--radix-evict-policy"}, "POLICY",
        string_format("which idle radix-cache leaf is evicted first when the reserved seq ids run out: lru, lfu, fifo, mru, filo, slru or "
                       "priority (the request's `priority` field: least important first). Ignored unless --radix-cache is set (default: %s)",
                       params.radix_evict_policy.c_str()),
        [](common_params & params, const std::string & value) {
            static const char * names[] = {"lru", "lfu", "fifo", "mru", "filo", "slru", "priority"};
            for (const char * n : names) {
                if (value == n) {
                    params.radix_evict_policy = value;
                    return;
                }
            }
            throw std::invalid_argument("error: invalid value for --radix-evict-policy (lru, lfu, fifo, mru, filo, slru, priority)\n");
        }
    ).set_env("LLAMA_ARG_RADIX_EVICT_POLICY").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--radix-evict-slru-hits"}, "N",
        string_format("with --radix-evict-policy slru: a tree node that this many prompts have passed through is protected from eviction "
                       "while an unprotected leaf exists (default: %d)", params.radix_evict_slru_hits),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("error: invalid value for --radix-evict-slru-hits (must be >= 1)\n");
            }
            params.radix_evict_slru_hits = value;
        }
    ).set_env("LLAMA_ARG_RADIX_EVICT_SLRU_HITS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--sched-policy"}, "POLICY",
        string_format("which waiting request starts when a slot frees up: fcfs (lowest `priority` value first, then arrival order) or lpm "
                       "(within one priority: the request with the longest prefix already cached first, and a request that shares a long prefix "
                       "with an earlier waiting one goes last so it can reuse that prefix) or spf (shortest-prefill-first: within one priority the request with the fewest uncached prompt tokens first, same hold-back rule) "
                       "or hrrn (highest response ratio next: within one priority the highest 1 + waited/uncached first, where waited = prefill tokens the server processed while the request waited; a request with nothing uncached goes first, same hold-back rule); lpm, spf and hrrn fall back to fcfs when more than 128 requests wait "
                       "(default: %s)", params.sched_policy.c_str()),
        [](common_params & params, const std::string & value) {
            if (value != "fcfs" && value != "lpm" && value != "spf" && value != "hrrn") {
                throw std::invalid_argument("error: invalid value for --sched-policy (fcfs, lpm, spf, hrrn)\n");
            }
            params.sched_policy = value;
        }
    ).set_env("LLAMA_ARG_SCHED_POLICY").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--sched-prefix-threshold"}, "N",
        string_format("with --sched-policy lpm: shared-prefix length in tokens above which a waiting request is held back behind an earlier "
                       "one sharing it, and below which a request's own cached prefix is compared with the waiting ones; -1 turns the "
                       "hold-back off (default: %d)", params.sched_prefix_threshold),
        [](common_params & params, int value) {
            if (value < -1) {
                throw std::invalid_argument("error: invalid value for --sched-prefix-threshold (must be >= -1)\n");
            }
            params.sched_prefix_threshold = value;
        }
    ).set_env("LLAMA_ARG_SCHED_PREFIX_THRESHOLD").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--context-shift"},
        {"--no-context-shift"},
        string_format("whether to use context shift on infinite text generation (default: %s)", params.ctx_shift ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.ctx_shift = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY}).set_env("LLAMA_ARG_CONTEXT_SHIFT"));
    add_opt(common_arg(
        {"--chunks"}, "N",
        string_format("max number of chunks to process (default: %d, -1 = all)", params.n_chunks),
        [](common_params & params, int value) {
            params.n_chunks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg({ "-fa", "--flash-attn" }, "[on|off|auto]",
                       // ADAPTED 2026-09-13 to this fork's `bool flash_attn` (common.h:761).
                       // The decision is already recorded and already implemented on the other
                       // side: common.cpp:2419-2423 keeps the bool and SYNTHESISES
                       // cparams.flash_attn_type from it, and every consumer in src/ reads the
                       // bool (llama.cpp:733, 771, 1082, 1287, 1520, 1856, ...). Nothing in this
                       // fork reads a three-valued flash-attn setting.
                       //
                       // 'auto' therefore maps to the fork's default, `true`. That is not a
                       // narrowing of behaviour: there is no deferred-decision path here for
                       // AUTO to select, so 'auto' and the default have always been the same
                       // run. The three spellings still parse, so no command line that worked
                       // before stops working.
                       string_format("set Flash Attention use ('on', 'off', or 'auto', default: '%s')",
                                     llama_flash_attn_type_name(params.flash_attn
                                         ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                         : LLAMA_FLASH_ATTN_TYPE_DISABLED)),
                       [](common_params & params, const std::string & value) {
                           if (is_truthy(value)) {
                               params.flash_attn = true;
                           } else if (is_falsey(value)) {
                               params.flash_attn = false;
                           } else if (is_autoy(value)) {
                               params.flash_attn = true;
                           } else {
                               throw std::runtime_error(
                                   string_format("error: unknown value for --flash-attn: '%s'\n", value.c_str()));
                           }
                       }).set_env("LLAMA_ARG_FLASH_ATTN"));
    add_opt(common_arg(
        {"-p", "--prompt"}, "PROMPT",
        "prompt to start generation with; for system message, use -sys",
        [](common_params & params, const std::string & value) {
            params.prompt = value;
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-sys", "--system-prompt"}, "PROMPT",
        "system prompt to use with model (if applicable, depending on chat template); llama-server: inserted as the\n"
        "first message of any chat request that carries no system message of its own",
        [](common_params & params, const std::string & value) {
            params.system_prompt = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_DIFFUSION, LLAMA_EXAMPLE_MTMD,
                    LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--perf"},
        {"--no-perf"},
        string_format("whether to enable internal libllama performance timings (default: %s)", params.no_perf ? "true" : "false"),
        [](common_params & params, bool value) {
            params.no_perf = !value;
            params.sampling.no_perf = !value;
        }
    ).set_env("LLAMA_ARG_PERF"));
    add_opt(common_arg(
        {"--show-timings"},
        {"--no-show-timings"},
        string_format("whether to show timing information after each response (default: %s)", params.show_timings ? "true" : "false"),
        [](common_params & params, bool value) {
            params.show_timings = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SHOW_TIMINGS"));
    add_opt(common_arg(
        {"-f", "--file"}, "FNAME",
        "a file containing the prompt (default: none)",
        [](common_params & params, const std::string & value) {
            params.prompt = read_file(value);
            // store the external file name in params
            params.prompt_file = value;
            if (!params.prompt.empty() && params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        // "-spf" is the fork's own spelling (pre-merge parser, common.cpp @ 35f0cac75:2675), where it set the
        // llama-server system prompt; restored 2026-09-27 together with the server consumer (server-common.cpp).
        {"-sysf", "-spf", "--system-prompt-file"}, "FNAME",
        "a file containing the system prompt (default: none); llama-server: see --system-prompt",
        [](common_params & params, const std::string & value) {
            params.system_prompt = read_file(value);
            if (!params.system_prompt.empty() && params.system_prompt.back() == '\n') {
                params.system_prompt.pop_back();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_DIFFUSION, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--in-file"}, "FNAME",
        "an input file (use comma-separated values to specify multiple files)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                std::ifstream file(item);
                if (!file) {
                    throw std::runtime_error(string_format("error: failed to open file '%s'\n", item.c_str()));
                }
                params.in_files.push_back(item);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"-bf", "--binary-file"}, "FNAME",
        "binary file containing the prompt (default: none)",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value, std::ios::binary);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            // store the external file name in params
            params.prompt_file = value;
            std::ostringstream ss;
            ss << file.rdbuf();
            params.prompt = ss.str();
            fprintf(stderr, "Read %zu bytes from binary file %s\n", params.prompt.size(), value.c_str());
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-e", "--escape"},
        {"--no-escape"},
        string_format("whether to process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)", params.escape ? "true" : "false"),
        [](common_params & params, bool value) {
            params.escape = value;
        }
    ));
    add_opt(common_arg(
        {"-ptc", "--print-token-count"}, "N",
        string_format("print token count every N tokens (default: %d)", params.n_print),
        [](common_params & params, int value) {
            params.n_print = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache"}, "FNAME",
        "file to cache prompt state for faster startup (default: none)",
        [](common_params & params, const std::string & value) {
            params.path_prompt_cache = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache-all"},
        "if specified, saves user input and generations to cache as well\n",
        [](common_params & params) {
            params.prompt_cache_all = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache-ro"},
        "if specified, uses the prompt cache but does not update it",
        [](common_params & params) {
            params.prompt_cache_ro = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-ld", "--logdir"}, "LOGDIR",
        "path under which to write a timestamped YAML run-log file after generation\n"
        "(default: unset -- no logging). Row S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-"
        "DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01: restores the fork's write_logfile()/"
        "yaml_dump_* mechanism (pre-merge common/common.cpp @ 35f0cac75:5282-5536), ported to"
        " tools/completion/completion.cpp against CURRENT common_params field names.",
        [](common_params & params, const std::string & value) {
            params.logdir = value;
            if (!params.logdir.empty() && params.logdir.back() != DIRECTORY_SEPARATOR) {
                params.logdir += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-r", "--reverse-prompt"}, "PROMPT",
        "halt generation at PROMPT, return control in interactive mode\n",
        [](common_params & params, const std::string & value) {
            params.antiprompt.emplace_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}));
    // SOURCE-READ: nexa-sdk cli/cmd/geniex/infer.go flags (BSD-3-Clause); behaviour only, no code copied
    add_opt(common_arg(
        {"--stop-file"}, "FNAME",
        "file of stop strings, one per line (blank lines skipped); each is appended as a reverse prompt",
        [](common_params & params, const std::string & value) {
            std::string text = read_file(value);
            size_t pos = 0;
            while (pos <= text.size()) {
                size_t end = text.find('\n', pos);
                if (end == std::string::npos) { end = text.size(); }
                std::string line = text.substr(pos, end - pos);
                if (!line.empty() && line.back() == '\r') { line.pop_back(); }
                if (!line.empty()) { params.antiprompt.emplace_back(line); }
                pos = end + 1;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--token-file"}, "FNAME",
        "file holding the prompt as whitespace-separated token ids (used instead of tokenizing the prompt text)",
        [](common_params & params, const std::string & value) {
            std::istringstream is(read_file(value));
            std::string tok;
            params.prompt_token_ids.clear();
            while (is >> tok) {
                try {
                    size_t used = 0;
                    long long id = std::stoll(tok, &used);
                    if (used != tok.size() || id < 0 || id > INT32_MAX) { throw std::invalid_argument("range"); }
                    params.prompt_token_ids.push_back((int32_t) id);
                } catch (const std::exception &) {
                    throw std::invalid_argument("--token-file: not a token id: " + tok);
                }
            }
            if (params.prompt_token_ids.empty()) {
                throw std::invalid_argument("--token-file: no token ids in " + value);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-sp", "--special"},
        string_format("special tokens output enabled (default: %s)", params.special ? "true" : "false"),
        [](common_params & params) {
            params.special = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-cnv", "--conversation"},
        {"-no-cnv", "--no-conversation"},
        "whether to run in conversation mode:\n"
        "- does not print special tokens and suffix/prefix\n"
        "- interactive mode is also enabled\n"
        "(default: auto enabled if chat template is available)",
        [](common_params & params, bool value) {
            params.conversation_mode = value ? COMMON_CONVERSATION_MODE_ENABLED : COMMON_CONVERSATION_MODE_DISABLED;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_DIFFUSION}));
    add_opt(common_arg(
        {"-st", "--single-turn"},
        "run conversation for a single turn only, then exit when done\n"
        "will not be interactive if first turn is predefined with --prompt\n"
        "(default: false)",
        [](common_params & params) {
            params.single_turn = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-i", "--interactive"},
        string_format("run in interactive mode (default: %s)", params.interactive ? "true" : "false"),
        [](common_params & params) {
            params.interactive = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-if", "--interactive-first"},
        string_format("run in interactive mode and wait for input right away (default: %s)", params.interactive_first ? "true" : "false"),
        [](common_params & params) {
            params.interactive_first = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-mli", "--multiline-input"},
        "allows you to write or paste multiple lines without ending each in '\\'",
        [](common_params & params) {
            params.multiline_input = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--in-prefix-bos"},
        "prefix BOS to user inputs, preceding the `--in-prefix` string",
        [](common_params & params) {
            params.input_prefix_bos = true;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--in-prefix"}, "STRING",
        "string to prefix user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_prefix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--in-suffix"}, "STRING",
        "string to suffix after user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_suffix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--warmup"},
        {"--no-warmup"},
        string_format("whether to perform warmup with an empty run (default: %s)", params.warmup ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.warmup = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--spm-infill"},
        string_format(
            "use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: %s)",
            params.spm_infill ? "enabled" : "disabled"
        ),
        [](common_params & params) {
            params.spm_infill = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--samplers"}, "SAMPLERS",
        string_format("samplers that will be used for generation in the order, separated by \';\'\n(default: %s)", sampler_type_names.c_str()),
        [](common_params & params, const std::string & value) {
            const auto sampler_names = string_split<std::string>(value, ';');
            // ADAPTED 2026-09-13, same reason as the samplers_sequence loop above. The fork's
            // parser takes allow_alt_names, which upstream's does not: passing true keeps
            // `top-k`/`typical`/`temp` working alongside the canonical names, so this is a
            // superset of what the upstream call accepted, not a narrowing.
            params.sampling.samplers_sequence = llama_sampling_types_from_names(sampler_names, true);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-s", "--seed"}, "SEED",
        string_format("RNG seed (default: %d, use random seed for %d)", params.sampling.seed, LLAMA_DEFAULT_SEED),
        [](common_params & params, const std::string & value) {
            params.sampling.seed = std::stoul(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--sampler-seq", "--sampling-seq"}, "SEQUENCE",
        string_format("simplified sequence for samplers that will be used (default: %s)", sampler_type_chars.c_str()),
        [](common_params & params, const std::string & value) {
            params.sampling.samplers_sequence = llama_sampling_types_from_chars(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--ignore-eos"},
        "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
        [](common_params & params) {
            params.sampling.ignore_eos = true;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--temp", "--temperature"}, "N",
        string_format("temperature (default: %.2f)", (double)params.sampling.temp),
        [](common_params & params, const std::string & value) {
            params.sampling.temp = std::stof(value);
            params.sampling.temp = std::max(params.sampling.temp, 0.0f);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
        }
    ).set_sampling().set_env("LLAMA_ARG_TEMPERATURE"));
    add_opt(common_arg(
        {"--top-k"}, "N",
        string_format("top-k sampling (default: %d, 0 = disabled)", params.sampling.top_k),
        [](common_params & params, int value) {
            params.sampling.top_k = value;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
        }
    ).set_sampling().set_env("LLAMA_ARG_TOP_K"));
    add_opt(common_arg(
        {"--top-p"}, "N",
        string_format("top-p sampling (default: %.2f, 1.0 = disabled)", (double)params.sampling.top_p),
        [](common_params & params, const std::string & value) {
            params.sampling.top_p = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P;
        }
    ).set_sampling().set_env("LLAMA_ARG_TOP_P"));
    add_opt(common_arg(
        {"--min-p"}, "N",
        string_format("min-p sampling (default: %.2f, 0.0 = disabled)", (double)params.sampling.min_p),
        [](common_params & params, const std::string & value) {
            params.sampling.min_p = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P;
        }
    ).set_sampling().set_env("LLAMA_ARG_MIN_P"));
    add_opt(common_arg(
        {"--top-nsigma", "--top-n-sigma"}, "N",
        string_format("top-n-sigma sampling (default: %.2f, -1.0 = disabled)", params.sampling.top_n_sigma),
        [](common_params & params, const std::string & value) {
            params.sampling.top_n_sigma = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--xtc-probability"}, "N",
        string_format("xtc probability (default: %.2f, 0.0 = disabled)", (double)params.sampling.xtc_probability),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_probability = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--xtc-threshold"}, "N",
        string_format("xtc threshold (default: %.2f, 1.0 = disabled)", (double)params.sampling.xtc_threshold),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_threshold = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--typical", "--typical-p"}, "N",
        string_format("locally typical sampling, parameter p (default: %.2f, 1.0 = disabled)", (double)params.sampling.typical_p),
        [](common_params & params, const std::string & value) {
            params.sampling.typical_p = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--repeat-last-n"}, "N",
        string_format("last n tokens to consider for penalize (default: %d, 0 = disabled)", params.sampling.penalty_last_n),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::runtime_error(string_format("error: invalid repeat-last-n = %d\n", value));
            }
            params.sampling.penalty_last_n = value;
            params.sampling.n_prev = std::max(params.sampling.n_prev, params.sampling.penalty_last_n);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--repeat-penalty"}, "N",
        string_format("penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)", (double)params.sampling.penalty_repeat),
        [](common_params & params, const std::string & value) {
            const float penalty_repeat = std::stof(value);
            if (!std::isfinite(penalty_repeat) ||
                penalty_repeat <= 0.0f ||
                !std::isfinite(1.0f/penalty_repeat)) {
                throw std::runtime_error("error: repeat-penalty must be finite and greater than 0\n");
            }
            params.sampling.penalty_repeat = penalty_repeat;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT;
        }
    ).set_sampling().set_env("LLAMA_ARG_REPEAT_PENALTY"));
    add_opt(common_arg(
        {"--presence-penalty"}, "N",
        string_format("repeat alpha presence penalty (default: %.2f, 0.0 = disabled)", (double)params.sampling.penalty_present),
        [](common_params & params, const std::string & value) {
            const float penalty_present = std::stof(value);
            if (!std::isfinite(penalty_present)) {
                throw std::runtime_error("error: presence-penalty must be finite\n");
            }
            params.sampling.penalty_present = penalty_present;
        }
    ).set_sampling().set_env("LLAMA_ARG_PRESENCE_PENALTY"));
    add_opt(common_arg(
        {"--penalize-prompt"},
        {"--no-penalize-prompt"},
        string_format("whether prompt tokens count towards the repeat, presence, frequency and DRY penalties (default: %s)",
                      params.sampling.penalize_prompt ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.sampling.penalize_prompt = value;
        }
    ).set_sampling().set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PENALIZE_PROMPT"));
    add_opt(common_arg(
        {"--frequency-penalty"}, "N",
        string_format("repeat alpha frequency penalty (default: %.2f, 0.0 = disabled)", (double)params.sampling.penalty_freq),
        [](common_params & params, const std::string & value) {
            const float penalty_freq = std::stof(value);
            if (!std::isfinite(penalty_freq)) {
                throw std::runtime_error("error: frequency-penalty must be finite\n");
            }
            params.sampling.penalty_freq = penalty_freq;
        }
    ).set_sampling().set_env("LLAMA_ARG_FREQUENCY_PENALTY"));
    add_opt(common_arg(
        {"--dry-multiplier"}, "N",
        string_format("set DRY sampling multiplier (default: %.2f, 0.0 = disabled)", (double)params.sampling.dry_multiplier),
        [](common_params & params, const std::string & value) {
            params.sampling.dry_multiplier = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-base"}, "N",
        string_format("set DRY sampling base value (default: %.2f)", (double)params.sampling.dry_base),
        [](common_params & params, const std::string & value) {
            float potential_base = std::stof(value);
            if (potential_base >= 1.0f)
            {
                params.sampling.dry_base = potential_base;
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-allowed-length"}, "N",
        string_format("set allowed length for DRY sampling (default: %d)", params.sampling.dry_allowed_length),
        [](common_params & params, int value) {
            params.sampling.dry_allowed_length = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-penalty-last-n"}, "N",
        string_format("set DRY penalty for the last n tokens (default: %d, 0 = disable, -1 = context size)", params.sampling.dry_penalty_last_n),
        [](common_params & params, int value) {
            // -1 = context size is ACCEPTED here, unlike upstream's current handler. Upstream dropped
            // it because its sampler clamps -1 to 0 and silently disables the penalty (ggml-org/
            // llama.cpp#25388, closed stale, never fixed). This fork's sampler implements it
            // (src/llama-sampling.cpp: `dry_penalty_last_n == -1 ? context_size : ...`) and -1 is the
            // default (common/sampling.h X-macro) -- rejecting it meant a user could not pass the
            // value the help text printed as the default.
            if (value < -1) {
                throw std::runtime_error(string_format("error: invalid dry-penalty-last-n = %d\n", value));
            }
            params.sampling.dry_penalty_last_n = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-sequence-breaker"}, "STRING",
        string_format("add sequence breaker for DRY sampling, clearing out default breakers (%s) in the process; use \"none\" to not use any sequence breakers\n",
            params.sampling.dry_sequence_breakers.empty() ? "none" :
            std::accumulate(std::next(params.sampling.dry_sequence_breakers.begin()),
                params.sampling.dry_sequence_breakers.end(),
                std::string("'") + (params.sampling.dry_sequence_breakers[0] == "\n" ? "\\n" : params.sampling.dry_sequence_breakers[0]) + "'",
                [](const std::string& a, const std::string& b) {
                    std::string formatted_b = (b == "\n") ? "\\n" : b;
                    return a + ", '" + formatted_b + "'";
                }).c_str()),
        [](common_params & params, const std::string & value) {
            static bool defaults_cleared = false;

            if (!defaults_cleared) {
                params.sampling.dry_sequence_breakers.clear();
                defaults_cleared = true;
            }

            if (value == "none") {
                params.sampling.dry_sequence_breakers.clear();
            } else {
                params.sampling.dry_sequence_breakers.emplace_back(value);
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--adaptive-target"}, "N",
        string_format("adaptive-p: select tokens near this probability (valid range 0.0 "
                      "to 1.0; negative = disabled) (default: %.2f)\n"
                      "[(more info)](https://github.com/ggml-org/llama.cpp/pull/17927)",
                      (double)params.sampling.adaptive_target),
        [](common_params & params, const std::string & value) {
            params.sampling.adaptive_target = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--adaptive-decay"}, "N",
        string_format("adaptive-p: decay rate for target adaptation over time. lower values "
                      "are more reactive, higher values are more stable.\n"
                      "(valid range 0.0 to 0.99) (default: %.2f)",
                      (double)params.sampling.adaptive_decay),
        [](common_params & params, const std::string & value) {
            params.sampling.adaptive_decay = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--spec-accept-relaxed-top-p"}, "P",
        string_format("LOSSY relaxed speculative acceptance: with temperature > 0 a draft token that lies inside the target's top-p nucleus of mass P "
                      "(0 < P < 1) is accepted whatever the probability ratio is; tokens outside it still go through exact rejection sampling. "
                      "Pulls the output toward the drafter; 0 = exact (default: %.2f)", (double)params.sampling.spec_relaxed_top_p),
        [](common_params & params, const std::string & value) {
            const float v = std::stof(value);
            if (!(v >= 0.0f && v < 1.0f)) {
                throw std::invalid_argument("error: --spec-accept-relaxed-top-p must be in [0, 1)");
            }
            params.sampling.spec_relaxed_top_p = v;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--spec-self-draft-type"}, "TYPE",
        "self-speculative decoding without a draft model: quantize the target GGUF to TYPE (q2_k, q3_k_m, q4_0, q4_k_s, q4_k_m, q5_0, q5_k_m, q6_k, q8_0, iq4_nl, iq4_xs) "
        "once, cache it next to the target as <name>.selfdraft-<TYPE>.gguf and use it as the draft model. An explicit -md wins; without --spec-type the plain "
        "draft stage is used. Pays off for F16/BF16/Q8_0 targets (a Q4-class target gains little from a lower-bit twin) (default: off)",
        [](common_params & params, const std::string & value) {
            params.speculative.self_draft_type = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_SELF_DRAFT_TYPE"));
    add_opt(common_arg(
        {"--sampler-topk-prefilter"},
        "build only the top-k candidates (exact, from the raw logits) instead of a vocabulary-sized array when top-k leads the sampler chain and "
        "nothing else needs the whole vocabulary (no penalties, grammar, DRY, mirostat, adaptive-p, FSD, logprobs, CFG); default: off",
        [](common_params & params) {
            params.sampling.topk_prefilter = true;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--fsd-alpha"}, "N",
        string_format("FSD (Frustratingly Simple Decoding, arXiv:2305.12675): deterministic anti-repetition decoding. "
                      "score = (1-alpha)*p - alpha*penalty, penalty = n-gram frequency of the candidate in prompt+output. "
                      "valid range 0.0 to 1.0; negative = disabled (default: %.2f)", (double)params.sampling.fsd_alpha),
        [](common_params & params, const std::string & value) {
            params.sampling.fsd_alpha = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--fsd-pool-size"}, "N",
        string_format("FSD: candidates that enter the softmax (default: %d)", params.sampling.fsd_pool_size),
        [](common_params & params, int value) {
            params.sampling.fsd_pool_size = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--fsd-k"}, "N",
        string_format("FSD: candidates kept after the softmax (default: %d)", params.sampling.fsd_k),
        [](common_params & params, int value) {
            params.sampling.fsd_k = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--fsd-n"}, "N",
        string_format("FSD: n-gram order of the anti-LM (default: %d)", params.sampling.fsd_n),
        [](common_params & params, int value) {
            params.sampling.fsd_n = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--fsd-beta"}, "N",
        string_format("FSD: back-off weight of the n-gram interpolation (default: %.2f)", (double)params.sampling.fsd_beta),
        [](common_params & params, const std::string & value) {
            params.sampling.fsd_beta = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--fsd-random-first-n"}, "N",
        string_format("Randomized FSD: for the first N sampling steps a coin flip chooses FSD or ordinary top-p/top-k sampling (default: %d = plain FSD)",
                      params.sampling.fsd_random_first_n),
        [](common_params & params, int value) {
            params.sampling.fsd_random_first_n = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--adaptive-p-untruncated"}, "{floor,refuse,allow}",
        "adaptive-p with no truncating sampler in front of it (no top-k/top-p/min-p/...) draws from the whole vocabulary.\n"
        "floor: apply min-p = --adaptive-p-floor first (default); refuse: server answers HTTP 400, CLI exits; allow: no change",
        [](common_params & params, const std::string & value) {
            const int32_t mode = common_adaptive_untruncated_from_name(value.c_str());
            if (mode < 0) {
                throw std::invalid_argument("error: --adaptive-p-untruncated must be floor, refuse or allow, got '" + value + "'");
            }
            params.sampling.adaptive_untruncated = mode;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--adaptive-p-floor"}, "N",
        string_format("min-p floor applied in front of adaptive-p when nothing else truncates (default: %.2f)",
                      (double)params.sampling.adaptive_floor),
        [](common_params & params, const std::string & value) {
            params.sampling.adaptive_floor = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--top-a"}, "N",
        string_format("top-a sampling: drop tokens with p < top_a * p_max^2 (default: %.2f, 0.0 = disabled)",
                      (double)params.sampling.top_a),
        [](common_params & params, const std::string & value) {
            params.sampling.top_a = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--epsilon-cutoff"}, "N",
        string_format("epsilon sampling (Hewitt et al. 2022): drop tokens with probability < N, a probability in [0,1) "
                      "(default: %.4f, 0.0 = disabled)", (double)params.sampling.epsilon_cutoff),
        [](common_params & params, const std::string & value) {
            params.sampling.epsilon_cutoff = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--eta-cutoff"}, "N",
        string_format("eta sampling (Hewitt et al. 2022): drop tokens with probability < min(N, sqrt(N)*exp(-entropy)) "
                      "(default: %.4f, 0.0 = disabled)", (double)params.sampling.eta_cutoff),
        [](common_params & params, const std::string & value) {
            params.sampling.eta_cutoff = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--no-repeat-ngram-size"}, "N",
        string_format("never generate a token that would repeat an N-gram already in the history, prompt included "
                      "(default: %d, 0 = disabled)", params.sampling.no_repeat_ngram_size),
        [](common_params & params, const std::string & value) {
            params.sampling.no_repeat_ngram_size = std::stoi(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--no-repeat-ngram-last-n"}, "N",
        string_format("history searched by --no-repeat-ngram-size (default: %d, -1 = context size)",
                      params.sampling.no_repeat_ngram_last_n),
        [](common_params & params, const std::string & value) {
            params.sampling.no_repeat_ngram_last_n = std::stoi(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--smoothing-factor"}, "N",
        string_format("quadratic (smooth) sampling strength, applied at the temperature step: near-tied top tokens keep "
                      "their odds while the tail is pushed down (default: %.2f, 0.0 = disabled)",
                      (double)params.sampling.smoothing_factor),
        [](common_params & params, const std::string & value) {
            params.sampling.smoothing_factor = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--repeat-penalty-slope"}, "N",
        string_format("repeat penalty for tokens seen only in the older half of the --repeat-last-n window: "
                      "1 + (repeat_penalty - 1) * slope; the newer half keeps the full penalty "
                      "(default: %.2f; 1.0 or <= 0 = uniform, as in koboldcpp)",
                      (double)params.sampling.penalty_slope),
        [](common_params & params, const std::string & value) {
            params.sampling.penalty_slope = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--repeat-decay"}, "N",
        string_format("penalty fade: past the --repeat-last-n window, repeat/frequency/presence penalties fall "
                      "linearly to 0 over N more tokens, so a token is not freed the moment it leaves the window "
                      "(ExLlamaV3 decay_range; default: %d, 0 = hard window edge)",
                      params.sampling.penalty_decay),
        [](common_params & params, const std::string & value) {
            const int v = std::stoi(value);
            if (v < 0) {
                throw std::invalid_argument("--repeat-decay must be >= 0");
            }
            params.sampling.penalty_decay = v;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--skew"}, "N",
        string_format("skew sampling: reshape the final random draw toward the most likely tokens (N < 0) or the less "
                      "likely ones (N > 0), over the candidates the other samplers kept (ExLlamaV2 / Aphrodite skew; "
                      "range -10 to 10; default: %.2f, 0 = off; no effect on greedy decoding)", (double)params.sampling.skew),
        [](common_params & params, const std::string & value) {
            const float v = std::stof(value);
            if (!(v >= -10.0f && v <= 10.0f)) {
                throw std::invalid_argument("--skew must be between -10 and 10");
            }
            params.sampling.skew = v;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--smoothing-curve"}, "N",
        string_format("smooth sampling curve: 1.0 = quadratic, other values add a cubic term (default: %.2f)",
                      (double)params.sampling.smoothing_curve),
        [](common_params & params, const std::string & value) {
            params.sampling.smoothing_curve = std::stof(value);
        }
    ).set_sampling());
    // Banned strings (ik_llama.cpp, MIT). The params survived the merge onto upstream's server but
    // the flags and every consumer did not; restored with the server's rewind-and-resample.
    add_opt(common_arg(
        {"--banned-string-file"}, "FNAME",
        "file with one banned string per line; when the output contains one, llama-server rewinds to "
        "where it starts, bans that token at that position and resamples (per-request field: banned_strings)",
        [](common_params & params, const std::string & value) {
            params.ban_phrases.clear();
            for (auto & str : string_split<std::string>(read_file(value), '\n')) {
                str.erase(std::remove(str.begin(), str.end(), '"'), str.end());
                if (!str.empty() && str.back() == '\r') {
                    str.pop_back();
                }
                if (!str.empty()) {
                    params.ban_phrases.push_back(str);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--banned-n"}, "N",
        string_format("tokens banned at the rewind position when a banned string is found: 1 = its first token, "
                      "-1 = every token of the match (default: %d)", params.banned_n),
        [](common_params & params, int value) {
            params.banned_n = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--dynatemp-range"}, "N",
        string_format("dynamic temperature range (default: %.2f, 0.0 = disabled)", (double)params.sampling.dynatemp_range),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_range = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dynatemp-exp"}, "N",
        string_format("dynamic temperature exponent (default: %.2f)", (double)params.sampling.dynatemp_exponent),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_exponent = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat"}, "N",
        string_format("use Mirostat sampling.\nTop K, Nucleus and Locally Typical samplers are ignored if used.\n"
        "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", params.sampling.mirostat),
        [](common_params & params, int value) {
            params.sampling.mirostat = value;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat-lr"}, "N",
        string_format("Mirostat learning rate, parameter eta (default: %.2f)", (double)params.sampling.mirostat_eta),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_eta = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat-ent"}, "N",
        string_format("Mirostat target entropy, parameter tau (default: %.2f)", (double)params.sampling.mirostat_tau),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_tau = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--allowed-token-ids"}, "ID[,ID...]",
        "restrict generation to these token ids ONLY; every other token is masked to -inf.\n"
        "repeatable, and accepts a comma-separated list, e.g. `--allowed-token-ids 15043,1234`.\n"
        "unlike --logit-bias this is a hard constraint, not a finite nudge",
        [](common_params & params, const std::string & value) {
            std::stringstream ss(value);
            std::string item;
            bool any = false;
            while (std::getline(ss, item, ',')) {
                // Trim: a list pasted from a config file routinely carries spaces.
                const size_t b = item.find_first_not_of(" \t");
                if (b == std::string::npos) {
                    continue;
                }
                const size_t e = item.find_last_not_of(" \t");
                item = item.substr(b, e - b + 1);
                try {
                    params.sampling.allowed_token_ids.push_back((llama_token) std::stoi(item));
                    any = true;
                } catch (const std::exception &) {
                    throw std::invalid_argument("invalid token id: " + item);
                }
            }
            // An argument that parsed to nothing is a user error, not an empty allow-list.
            // Accepting it silently would look identical to not passing the flag at all.
            if (!any) {
                throw std::invalid_argument("no token ids parsed from: " + value);
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--suppress-sequence"}, "ID[,ID...]",
        "ban a TOKEN sequence (CTranslate2 suppress_sequences): a one-token sequence is banned everywhere; a longer one bans its LAST token\n"
        "whenever the prompt + output so far ends with all the tokens before it. Repeatable, one sequence per occurrence,\n"
        "e.g. `--suppress-sequence 15043,1234 --suppress-sequence 7`. Unlike --banned-strings no text is generated and rewound",
        [](common_params & params, const std::string & value) {
            std::stringstream ss(value);
            std::string item;
            std::vector<llama_token> seq;
            while (std::getline(ss, item, ',')) {
                const size_t b = item.find_first_not_of(" \t");
                if (b == std::string::npos) {
                    continue;
                }
                item = item.substr(b, item.find_last_not_of(" \t") - b + 1);
                try {
                    seq.push_back((llama_token) std::stoi(item));
                } catch (const std::exception &) {
                    throw std::invalid_argument("invalid token id: " + item);
                }
            }
            if (seq.empty()) {
                throw std::invalid_argument("no token ids parsed from: " + value);
            }
            params.sampling.suppress_sequences.push_back(std::move(seq));
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--watermark-key"}, "KEY",
        "secret 64-bit key for the Gumbel-max text watermark (decimal or 0x hex). Server-side only: a\n"
        "request can switch the watermark on or off but can never read or set the key. Prefer the env\n"
        "var over the flag, since a command line is visible in the process list",
        [](common_params & params, const std::string & value) {
            if (value.empty() || value[0] == '-') {
                throw std::invalid_argument("watermark key must be a non-negative 64-bit integer: " + value);
            }
            size_t used = 0;
            unsigned long long key = 0;
            try {
                key = std::stoull(value, &used, 0);
            } catch (const std::exception &) {
                throw std::invalid_argument("watermark key must be a non-negative 64-bit integer: " + value);
            }
            if (used != value.size()) {
                throw std::invalid_argument("trailing characters in watermark key: " + value);
            }
            params.sampling.watermark_key     = (uint64_t) key;
            params.sampling.watermark_key_set = true;
        }
    ).set_env("LLAMA_ARG_WATERMARK_KEY").set_sampling());
    add_opt(common_arg(
        {"--watermark"},
        "watermark generated text by default (needs --watermark-key). A request's \"watermark\" field\n"
        "overrides this per request. Greedy decoding (temp <= 0) is never watermarked",
        [](common_params & params) {
            params.sampling.watermark = true;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--watermark-context-width"}, "N",
        string_format("number of previous generated tokens that seed the watermark noise (default: %d)",
                      params.sampling.watermark_context_width),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("--watermark-context-width must be >= 1");
            }
            params.sampling.watermark_context_width = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--watermark-dedup-history"}, "N",
        string_format("generated positions searched for a repeated watermark context; a repeat is sampled\n"
                      "without the watermark. <= 0 searches all (default: %d)",
                      params.sampling.watermark_dedup_history),
        [](common_params & params, int value) {
            params.sampling.watermark_dedup_history = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--watermark-scheme"}, "{gumbel,kgw,synthid}",
        "watermark scheme (default: gumbel). gumbel = keyed Gumbel-max, vLLM-compatible, leaves each token's\n"
        "distribution unchanged; kgw = TGI's green list (Kirchenbauer et al.), biases green tokens by\n"
        "--watermark-kgw-delta; synthid = SynthID-Text tournament sampling. A request's \"watermark_scheme\"\n"
        "overrides this. All use --watermark-key; /watermark/detect takes the same \"scheme\"",
        [](common_params & params, const std::string & value) {
            const int32_t s = common_watermark::scheme_from_name(value.c_str());
            if (s < 0) {
                throw std::invalid_argument("--watermark-scheme must be gumbel, kgw or synthid: " + value);
            }
            params.sampling.watermark_scheme = s;
        }
    ).set_env("LLAMA_ARG_WATERMARK_SCHEME").set_sampling());
    add_opt(common_arg(
        {"--watermark-kgw-gamma"}, "F",
        string_format("kgw: fraction of the vocabulary that is green at each position, in (0,1) (default: %.2f)",
                      (double) params.sampling.watermark_kgw_gamma),
        [](common_params & params, const std::string & value) {
            const float v = std::stof(value);
            if (!(v > 0.0f && v < 1.0f)) {
                throw std::invalid_argument("--watermark-kgw-gamma must be in (0,1)");
            }
            params.sampling.watermark_kgw_gamma = v;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--watermark-kgw-delta"}, "F",
        string_format("kgw: logit bias added to green tokens, > 0 (default: %.2f)",
                      (double) params.sampling.watermark_kgw_delta),
        [](common_params & params, const std::string & value) {
            const float v = std::stof(value);
            if (!(v > 0.0f)) {
                throw std::invalid_argument("--watermark-kgw-delta must be > 0");
            }
            params.sampling.watermark_kgw_delta = v;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--watermark-synthid-depth"}, "N",
        string_format("synthid: tournament layers, 1..64 (default: %d)", params.sampling.watermark_synthid_depth),
        [](common_params & params, int value) {
            if (value < 1 || value > 64) {
                throw std::invalid_argument("--watermark-synthid-depth must be in [1, 64]");
            }
            params.sampling.watermark_synthid_depth = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-l", "--logit-bias"}, "TOKEN_ID(+/-)BIAS",
        "modifies the likelihood of token appearing in the completion,\n"
        "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
        "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'",
        [](common_params & params, const std::string & value) {
            std::stringstream ss(value);
            llama_token key;
            char sign;
            std::string value_str;
            try {
                if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                    const float bias = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                    params.sampling.logit_bias.push_back({key, bias});
                } else {
                    throw std::invalid_argument("invalid input format");
                }
            } catch (const std::exception&) {
                throw std::invalid_argument("invalid input format");
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--grammar", "--grammar-string"}, "GRAMMAR",
        "BNF-like grammar to constrain generations (see samples in grammars/ dir)",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, value};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--grammar-file", "--grammar-path"}, "FNAME",
        "file to read grammar from",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, read_file(value)};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-j", "--json-schema"}, "SCHEMA",
        "JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(json::parse(value))};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-jf", "--json-schema-file"}, "FILE",
        "File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string schema;
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(schema)
            );
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(json::parse(schema))};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-bs", "--backend-sampling"},
        "enable backend sampling (experimental) (default: disabled)",
        [](common_params & params) {
            params.sampling.backend_sampling = true;
        }
    ).set_sampling().set_env("LLAMA_ARG_BACKEND_SAMPLING"));
    add_opt(common_arg(
        {"--pooling"}, "{none,mean,cls,last,rank,attention}",
        "pooling type for embeddings, use model default if unspecified",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "none") { params.pooling_type = LLAMA_POOLING_TYPE_NONE; }
            else if (value == "mean") { params.pooling_type = LLAMA_POOLING_TYPE_MEAN; }
            else if (value == "cls")  { params.pooling_type = LLAMA_POOLING_TYPE_CLS;  }
            else if (value == "last") { params.pooling_type = LLAMA_POOLING_TYPE_LAST; }
            else if (value == "rank") { params.pooling_type = LLAMA_POOLING_TYPE_RANK; }
            else if (value == "attention") { params.pooling_type = LLAMA_POOLING_TYPE_ATTENTION; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_DEBUG}).set_env("LLAMA_ARG_POOLING"));
    add_opt(common_arg(
        {"--attention"}, "{causal,non-causal}",
        "attention type for embeddings, use model default if unspecified",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "causal") { params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL; }
            else if (value == "non-causal") { params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--rope-scaling"}, "{none,linear,yarn}",
        "RoPE frequency scaling method, defaults to linear unless specified by the model",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "none") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
            else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
            else if (value == "yarn") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_ROPE_SCALING_TYPE"));
    add_opt(common_arg(
        {"--rope-scale"}, "N",
        "RoPE context scaling factor, expands context by a factor of N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = 1.0f / std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_SCALE"));
    add_opt(common_arg(
        {"--rope-freq-base"}, "N",
        "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
        [](common_params & params, const std::string & value) {
            params.rope_freq_base = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_BASE"));
    add_opt(common_arg(
        {"--rope-freq-scale"}, "N",
        "RoPE frequency scaling factor, expands context by a factor of 1/N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_SCALE"));
    add_opt(common_arg(
        {"--yarn-orig-ctx"}, "N",
        string_format("YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx),
        [](common_params & params, int value) {
            params.yarn_orig_ctx = value;
        }
    ).set_env("LLAMA_ARG_YARN_ORIG_CTX"));
    add_opt(common_arg(
        {"--yarn-ext-factor"}, "N",
        string_format("YaRN: extrapolation mix factor (default: %.2f, 0.0 = full interpolation)", (double)params.yarn_ext_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_ext_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_EXT_FACTOR"));
    add_opt(common_arg(
        {"--yarn-attn-factor"}, "N",
        string_format("YaRN: scale sqrt(t) or attention magnitude (default: %.2f)", (double)params.yarn_attn_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_attn_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_ATTN_FACTOR"));
    add_opt(common_arg(
        {"--yarn-beta-slow"}, "N",
        string_format("YaRN: high correction dim or alpha (default: %.2f)", (double)params.yarn_beta_slow),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_slow = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_SLOW"));
    add_opt(common_arg(
        {"--yarn-beta-fast"}, "N",
        string_format("YaRN: low correction dim or beta (default: %.2f)", (double)params.yarn_beta_fast),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_fast = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_FAST"));
    add_opt(common_arg(
        {"-gan", "--grp-attn-n"}, "N",
        string_format("group-attention factor (default: %d)", params.grp_attn_n),
        [](common_params & params, int value) {
            params.grp_attn_n = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_N").set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_PASSKEY}));
    add_opt(common_arg(
        {"-gaw", "--grp-attn-w"}, "N",
        string_format("group-attention width (default: %d)", params.grp_attn_w),
        [](common_params & params, int value) {
            params.grp_attn_w = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_W").set_examples({LLAMA_EXAMPLE_COMPLETION}));
    // RESTORED (S69-EXAMPLES-KV-VIEW-DISPOSITION-2026-09-08-01): params.dump_kv_cache and its
    // driving CLI flag were both missing even though common.h already declared (now also
    // defines) llama_kv_cache_dump_view_seqs and examples/parallel/parallel.cpp reads
    // params.dump_kv_cache directly -- see common.h's own comment on the field/functions for
    // the full account.
    add_opt(common_arg(
        {"-dkvc", "--dump-kv-cache"},
        "verbose print of the KV cache",
        [](common_params & params) {
            params.dump_kv_cache = true;
        }
    // NOTE: llama_example has no LLAMA_EXAMPLE_LOOKAHEAD tag (checked common.h's enum) --
    // examples/lookahead/lookahead.cpp itself parses under LLAMA_EXAMPLE_COMMON, so this flag
    // is scoped to COMMON (covers lookahead) plus the 2 tags that do exist for the other two.
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-kvo", "--kv-offload"},
        {"-nkvo", "--no-kv-offload"},
        string_format("whether to enable KV cache offloading (default: %s)", params.no_kv_offload ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_kv_offload = !value;
        }
    ).set_env("LLAMA_ARG_KV_OFFLOAD"));
    add_opt(common_arg(
        {"--repack"},
        {"-nr", "--no-repack"},
        string_format("whether to enable weight repacking (default: %s)", params.no_extra_bufts ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_extra_bufts = !value;
        }
    ).set_env("LLAMA_ARG_REPACK"));
    add_opt(common_arg(
        {"--no-host"},
        "bypass host buffer allowing extra buffers to be used",
        [](common_params & params) {
            params.no_host = true;
        }
    ).set_env("LLAMA_ARG_NO_HOST"));
    add_opt(common_arg(
        {"--k-smooth"}, "ALPHA",
        "K smoothing for a quantized K cache (QServe SmoothAttention, scales derived from the weights -- no\n"
        "calibration file). Scales K down and Q up per channel before RoPE: q.k is unchanged, the K cache\n"
        "quantizes better. 0 = off (default); 0.5 = the measured value. NORM/NEOX rope, non-QK-norm layers only;\n"
        "the load log says how many layers qualified. Measured on qwen2.5-0.5b (q8_0 K cache): PPL loss +1.86 -> +0.07",
        [](common_params & params, const std::string & value) {
            const float a = std::stof(value);
            if (!(a >= 0.0f && a <= 1.0f)) {
                throw std::invalid_argument("--k-smooth: ALPHA must be in [0, 1]");
            }
            params.k_smooth_alpha = a;
        }
    ).set_env("LLAMA_ARG_K_SMOOTH"));
    add_opt(common_arg(
        {"--nvfp4-precision"}, "{auto,w4a4,int8}",
        "activation precision for NVFP4_KS weights (needs LLAMA_NVFP4_IMPORT_KS=1 to import NVFP4 checkpoints):\n"
        "auto = use the checkpoint .input_scale when valid (default); w4a4 = always dynamic activation scale;\n"
        "int8 = never W4A4",
        [](common_params & params, const std::string & value) {
            if      (value == "auto") { params.nvfp4_precision = LLAMA_NVFP4_PRECISION_AUTO; }
            else if (value == "w4a4") { params.nvfp4_precision = LLAMA_NVFP4_PRECISION_W4A4; }
            else if (value == "int8") { params.nvfp4_precision = LLAMA_NVFP4_PRECISION_INT8; }
            else { throw std::invalid_argument("--nvfp4-precision: expected auto, w4a4 or int8"); }
        }
    ).set_env("LLAMA_ARG_NVFP4_PRECISION"));

    add_opt(common_arg(
        {"-ctk", "--cache-type-k"}, "TYPE",
        string_format(
            "KV cache data type for K\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            // ADAPTED (S69-ARGCPP-MODEL-NESTED-VS-FLAT-2026-09-08-01): common_params::
            // cache_type_k is std::string, not ggml_type -- same pattern as
            // params.speculative.cache_type_k above; conversion happens at point of use
            // (common_model_params_to_llama/common_context_params_to_llama, common.cpp).
            params.cache_type_k.empty() ? "f16" : params.cache_type_k.c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_k = value;
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_K"));
    add_opt(common_arg(
        {"-ctv", "--cache-type-v"}, "TYPE",
        string_format(
            "KV cache data type for V\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            params.cache_type_v.empty() ? "f16" : params.cache_type_v.c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_v = value;
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_V"));
    add_opt(common_arg(
        {"--hellaswag"},
        "compute HellaSwag score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.hellaswag = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--hellaswag-tasks"}, "N",
        string_format("number of tasks to use when computing the HellaSwag score (default: %zu)", params.hellaswag_tasks),
        [](common_params & params, int value) {
            params.hellaswag_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--winogrande"},
        "compute Winogrande score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.winogrande = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--winogrande-tasks"}, "N",
        string_format("number of tasks to use when computing the Winogrande score (default: %zu)", params.winogrande_tasks),
        [](common_params & params, int value) {
            params.winogrande_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--multiple-choice"},
        "compute multiple choice score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.multiple_choice = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--multiple-choice-tasks"}, "N",
        string_format("number of tasks to use when computing the multiple choice score (default: %zu)", params.multiple_choice_tasks),
        [](common_params & params, int value) {
            params.multiple_choice_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--kl-divergence"},
        "computes KL-divergence to logits provided via --kl-divergence-base",
        [](common_params & params) {
            params.kl_divergence = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--save-all-logits", "--kl-divergence-base"}, "FNAME",
        "set logits file",
        [](common_params & params, const std::string & value) {
            params.logits_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--ppl-stride"}, "N",
        string_format("stride for perplexity calculation (default: %d)", params.ppl_stride),
        [](common_params & params, int value) {
            params.ppl_stride = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    // Behaviour of Thireus/ik_llama.cpp branch th/perplexity_with_split_on (3250761b1..a946d1c0b, MIT), re-implemented.
    add_opt(common_arg(
        {"--split-on"}, "STRING",
        "split the input text on this exact string and evaluate every non-empty piece as its own sequence "
        "(pieces longer than the context are truncated). perplexity: each piece is scored; not combinable with "
        "--ppl-stride or --kl-divergence-base. imatrix: activations are collected over each piece "
        "(Thireus th/imatrix_with_split_on)",
        [](common_params & params, const std::string & value) {
            params.ppl_split_on = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--verbose-chunks"},
        "with --split-on: print each piece's text to stdout before it is evaluated",
        [](common_params & params) {
            params.ppl_verbose_chunks = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--ppl-output-type"}, "<0|1>",
        string_format("output type for perplexity calculation (default: %d)", params.ppl_output_type),
        [](common_params & params, int value) {
            params.ppl_output_type = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"-dt", "--defrag-thold"}, "N",
        string_format("KV cache defragmentation threshold (DEPRECATED)"),
        [](common_params & params, const std::string & value) {
            GGML_UNUSED(params);
            GGML_UNUSED(value);
            LOG_WRN("DEPRECATED: --defrag-thold is deprecated and no longer necessary to specify\n");
        }
    ).set_env("LLAMA_ARG_DEFRAG_THOLD"));
    if (ex == LLAMA_EXAMPLE_SERVER) {
        // this is to make sure this option appears in the server-specific section of the help message
        add_opt(common_arg(
            {"-np", "--parallel"}, "N",
            string_format("number of server slots (default: %d, -1 = auto)", params.n_parallel),
            [](common_params & params, int value) {
                if (value == 0) {
                    throw std::invalid_argument("error: invalid value for n_parallel\n");
                }
                params.n_parallel = value;
            }
        ).set_env("LLAMA_ARG_N_PARALLEL").set_examples({LLAMA_EXAMPLE_SERVER}));
    } else {
        add_opt(common_arg(
            {"-np", "--parallel"}, "N",
            string_format("number of parallel sequences to decode (default: %d)", params.n_parallel),
            [](common_params & params, int value) {
                params.n_parallel = value;
            }
        ).set_env("LLAMA_ARG_N_PARALLEL"));
    }
    // DISABLE-REVERSIBLE: --cfg-guidance-slots (default 0) is the named switch itself -- this IS
    // the opt-in flag, not code disabling a working feature. Row
    // S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01 (RESEARCH-DONE: store.get, read in
    // full before this flag was designed) specifies it opt-in precisely so a server that never
    // sets it keeps n_seq_max unchanged (true no-op, no extra memory) -- the re-enable condition is
    // simply passing N > 0.
    // RESEARCH-DONE: src/llama-context.cpp:516-530 (SOURCE-READ this turn, coordinator review
    // 2026-09-27) -- kv_unified=false (this fork's default, common.h:866) divides n_ctx evenly
    // across n_seq_max, so the help text below states that cost rather than hiding it.
    add_opt(common_arg(
        {"--cfg-guidance-slots"}, "N",
        string_format("number of extra sequence ids reserved for per-slot classifier-free guidance, "
                       "on top of --parallel (default: %d = disabled). Row "
                       "S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01. 0 keeps n_seq_max "
                       "identical to a build without this flag (no extra memory); a request setting "
                       "cfg_scale/negative_prompt is refused with an error unless this is > 0. "
                       "COST: with the default --kv-unified false, n_ctx is split evenly across "
                       "n_seq_max, so N>0 SHRINKS every normal slot's own context "
                       "(n_ctx_slot ~= n_ctx / (--parallel + N) instead of n_ctx / --parallel); pass "
                       "--kv-unified to share one unsplit cache instead and avoid that shrink. Only "
                       "the safe subset of requests may use guidance: no speculative decoding, no "
                       "banned_strings/bad_words, no prompt-checkpointing (--ctx-checkpoints 0, or a "
                       "model/SWA setup that never checkpoints), and --no-context-shift",
                       params.n_cfg_guidance_slots),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("error: invalid value for --cfg-guidance-slots (must be >= 0)\n");
            }
            params.n_cfg_guidance_slots = value;
        }
    ).set_env("LLAMA_ARG_CFG_GUIDANCE_SLOTS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--decision-seqs"}, "N",
        string_format("sequence ids reserved for the /v1/decision endpoint (parallel constrained decisions among a finite set of "
                      "allowed values, tools/parallel-decision), on top of --parallel, --cfg-guidance-slots and the radix cache's ids; "
                      "enables the endpoint and forces --kv-unified (default: %d = disabled, minimum 3: cached prefix, trunk, branch)",
                      params.n_seq_decision),
        [](common_params & params, int value) {
            if (value != 0 && value < 3) {
                throw std::invalid_argument("error: --decision-seqs needs at least 3 (cached prefix, trunk, one branch), or 0 to disable\n");
            }
            params.n_seq_decision = value;
        }
    ).set_env("LLAMA_ARG_DECISION_SEQS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-ns", "--sequences"}, "N",
        string_format("number of sequences to decode (default: %d)", params.n_sequences),
        [](common_params & params, int value) {
            params.n_sequences = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-cb", "--cont-batching"},
        {"-nocb", "--no-cont-batching"},
        string_format("whether to enable continuous batching (a.k.a dynamic batching) (default: %s)", params.cont_batching ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.cont_batching = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CONT_BATCHING"));
    add_opt(common_arg(
        {"-mm", "--mmproj"}, "FILE",
        "path to a multimodal projector file. see tools/mtmd/README.md\n"
        "note: if -hf is used, this argument can be omitted",
        [](common_params & params, const std::string & value) {
            params.mmproj.path = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ"));
    add_opt(common_arg(
        {"-mmu", "--mmproj-url"}, "URL",
        "URL to a multimodal projector file. see tools/mtmd/README.md",
        [](common_params & params, const std::string & value) {
            params.mmproj.url = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ_URL"));
    add_opt(common_arg(
        {"--mmproj-auto"},
        {"--no-mmproj", "--no-mmproj-auto"},
        string_format("whether to use multimodal projector file (if available), useful when using -hf (default: %s)", params.no_mmproj ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_mmproj = !value;
        }
    ).set_examples({LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_DOWNLOAD}).set_env("LLAMA_ARG_MMPROJ_AUTO"));
    add_opt(common_arg(
        {"--mmproj-offload"},
        {"--no-mmproj-offload"},
        string_format("whether to enable GPU offloading for multimodal projector (default: %s)", params.mmproj_use_gpu ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.mmproj_use_gpu = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ_OFFLOAD"));
    add_opt(common_arg(
        // note: "-mmdev" must sort after "--rpc" in the preset map, else RPC devices are not registered yet
        {"-mmdev", "--mmproj-device"}, "DEVICE",
        "device to use for multimodal projector (none = don't offload, default: follows --device)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            if (value == "none") {
                params.mmproj_use_gpu = false;
                params.mmproj_device  = nullptr;
                return;
            }
            auto devices = parse_device_list(value);
            // parse_device_list pushes nullptr at back so devices is length 2 for single device.
            if (devices.size() > 2) {
                throw std::invalid_argument("only one device may be specified for mmproj");
            }
            params.mmproj_use_gpu = true;
            params.mmproj_device  = devices.front();
        }
    ).set_examples(mmproj_examples).set_env("MTMD_BACKEND_DEVICE")); // no LLAMA_ARG_ prefix for backward compatibility reason
    add_opt(common_arg(
        {"--image", "--audio", "--video"}, "FILE",
        "path to an image, audio, or video file. use with multimodal models, use comma-separated values for multiple files\n",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                params.image.emplace_back(item);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--image-min-tokens"}, "N",
        "minimum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)",
        [](common_params & params, int value) {
            params.image_min_tokens = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_IMAGE_MIN_TOKENS"));
    add_opt(common_arg(
        {"--image-max-tokens"}, "N",
        "maximum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)",
        [](common_params & params, int value) {
            params.image_max_tokens = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_IMAGE_MAX_TOKENS"));
    add_opt(common_arg(
        {"--mtmd-batch-max-tokens"}, "N",
        string_format("maximum number of image tokens per batch when encoding images (default: %d)", params.mtmd_batch_max_tokens),
        [](common_params & params, int value) {
            params.mtmd_batch_max_tokens = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MTMD_BATCH_MAX_TOKENS"));
    // SOURCE-READ: upstream llama.cpp f29551215 (MIT, #24318) common/arg.cpp:2647-2667. Upstream registers these for
    // mmproj_examples, which includes TTS; tools/tts never decodes video, so TTS is left out here. llama-cli is kept:
    // it runs llama_server() in-process (tools/cli/cli-server.h:47), which applies them (tools/server/server.cpp).
    add_opt(common_arg(
        {"--video-fps"}, "N",
        string_format("target video frame rate; <= 0 uses the clip's native rate (default: %.1f)", params.video_fps),
        [](common_params & params, const std::string & value) {
            params.video_fps = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_VIDEO_FPS"));
    add_opt(common_arg(
        {"--video-timestamp-interval"}, "N",
        string_format("interval in milliseconds between text timestamps; <= 0 means no timestamps (default: %" PRId64 ")", params.video_timestamp_interval_ms),
        [](common_params & params, const std::string & value) {
            params.video_timestamp_interval_ms = (int64_t) std::stoll(value);
        }
    ).set_examples({LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_VIDEO_TIMESTAMP_INTERVAL"));
    add_opt(common_arg(
        {"--video-ffmpeg-dir"}, "DIR",
        "path to the directory containing ffmpeg and ffprobe (default: search in PATH)",
        [](common_params & params, const std::string & value) {
            params.video_ffmpeg_bin_dir = value;
        }
    ).set_examples({LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_VIDEO_FFMPEG_DIR"));
    add_opt(common_arg(
        {"--rpc"}, "SERVERS",
        "comma-separated list of RPC servers (host:port)",
        [](common_params & params, const std::string & value) {
            // upstream #11262 made --rpc registry-only (add_rpc_devices) and dropped this assignment; the fork's model
            // loader never consults the registry for placement -- it reads mparams.rpc_servers (common.cpp) -- so after
            // that merge --rpc registered a device that nothing used and every layer stayed local (measured: has_rpc
            // false, "offloaded 0/31 layers", no RPC buffers). Keep both: the registry entry serves --list-devices /
            // -mmdev, the string drives the fork's placement.
            params.rpc_servers = value;
            if (!llama_supports_rpc()) {
                throw std::invalid_argument("RPC not supported in this build");
            }
            add_rpc_devices(value);
            GGML_UNUSED(params);
        }
    ).set_env("LLAMA_ARG_RPC"));
    // fork: the pre --load-mode spellings stay registered (scripts pass them), as DEPRECATED aliases onto the same load_mode
    // (restored 2026-11-05 from HEAD common/arg.cpp:4572-4596; upstream removed them in favour of --load-mode)
    add_opt(common_arg(
        {"--mlock"},
        "DEPRECATED in favor of `--load-mode`: force system to keep model in RAM rather than swapping or compressing",
        [](common_params & params) {
            LOG_WRN("DEPRECATED: --mlock is deprecated. use --load-mode mlock instead\n");
            params.load_mode = LLAMA_LOAD_MODE_MLOCK;
        }
    ).set_env("LLAMA_ARG_MLOCK"));
    add_opt(common_arg(
        {"--mmap"},
        {"--no-mmap"},
        "DEPRECATED in favor of `--load-mode`: whether to memory-map model. (if mmap disabled, slower load but may reduce pageouts if not using mlock)",
        [](common_params & params, bool value) {
            LOG_WRN("DEPRECATED: --mmap and --no-mmap are deprecated. use --load-mode mmap instead\n");
            params.load_mode = value ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        }
    ).set_env("LLAMA_ARG_MMAP"));
    add_opt(common_arg(
        {"-dio", "--direct-io"},
        {"-ndio", "--no-direct-io"},
        "DEPRECATED in favor of `--load-mode`: use DirectIO if available",
        [](common_params & params, bool value) {
            LOG_WRN("DEPRECATED: --direct-io and --no-direct-io are deprecated. use --load-mode dio instead\n");
            params.load_mode = value ? LLAMA_LOAD_MODE_DIRECT_IO : LLAMA_LOAD_MODE_NONE;
        }
    ).set_env("LLAMA_ARG_DIO"));
    add_opt(common_arg(
        {"-lm", "--load-mode"}, "MODE",
        "model loading mode (default: auto)\n"
        "- auto: mmap, unless a device does not support it\n"
        "- none: no special loading mode\n"
        "- mmap: memory-map model (if mmap disabled, slower load but may reduce pageouts if not using mlock)\n"
        "- mlock: force system to keep model in RAM rather than swapping or compressing\n"
        "- mmap+mlock: mmap + force system to keep model in RAM rather than swapping or compressing\n"
        "- dio: use DirectIO if available\n",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "auto")       { params.load_mode = LLAMA_LOAD_MODE_AUTO;       }
            else if (value == "none")       { params.load_mode = LLAMA_LOAD_MODE_NONE;       }
            else if (value == "mmap")       { params.load_mode = LLAMA_LOAD_MODE_MMAP;       }
            else if (value == "mlock")      { params.load_mode = LLAMA_LOAD_MODE_MLOCK;      }
            else if (value == "mmap+mlock") { params.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK; }
            else if (value == "dio")        { params.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;  }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_LOAD_MODE"));
    add_opt(common_arg(
        {"-lzm", "--lazy-mode"}, "MODE",
        "on-demand reading of certain tensors, for example per-layer embeddings (default: auto)\n"
        "- on: read the rows of such tensors from disk on demand instead of keeping them resident (requires mmap; = --defer-ple)\n"
        "- auto: on, but only for tensors larger than 4 GiB\n"
        "- off: always keep them resident",
        [](common_params & params, const std::string & value) {
            // upstream's field AND the defer_ple pair this fork's loader reads (src/llama.cpp llama_model_load)
            /**/ if (value == "on")   { params.lazy_mode = LLAMA_LAZY_MODE_ON;   params.defer_ple = true;  params.defer_ple_auto = false; }
            else if (value == "auto") { params.lazy_mode = LLAMA_LAZY_MODE_AUTO; params.defer_ple = false; params.defer_ple_auto = true;  }
            else if (value == "off")  { params.lazy_mode = LLAMA_LAZY_MODE_OFF;  params.defer_ple = false; params.defer_ple_auto = false; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_LAZY_MODE"));
    add_opt(common_arg(
        {"--numa"}, "TYPE",
        "attempt optimizations that help on some NUMA systems\n"
        "- distribute: spread execution evenly over all nodes\n"
        "- isolate: only spawn threads on CPUs on the node that execution started on\n"
        "- numactl: use the CPU map provided by numactl\n"
        "if run without this previously, it is recommended to drop the system page cache before using this\n"
        "see https://github.com/ggml-org/llama.cpp/issues/1437",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "distribute" || value == "") { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
            else if (value == "isolate") { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
            else if (value == "numactl") { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_NUMA"));
    add_opt(common_arg(
        {"-dev", "--device"}, "<dev1,dev2,..>",
        "comma-separated list of devices to use for offloading (none = don't offload)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            // ADAPTED (S69-ARGCPP-MODEL-NESTED-VS-FLAT-2026-09-08-01): top-level
            // common_params::devices is a flat std::string (raw CSV) here, same shape/reason
            // as params.speculative.devices above.
            params.devices = value;
        }
    ).set_env("LLAMA_ARG_DEVICE"));
    add_opt(common_arg(
        {"--list-devices"},
        "print list of available devices and exit",
        [](common_params &) {
            common_print_available_devices();
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"-ot", "--override-tensor"}, "<tensor name pattern>=<buffer type>,...",
        "override tensor buffer type", [](common_params & params, const std::string & value) {
            parse_tensor_buffer_overrides(value, params.tensor_buft_overrides);
        }
    ).set_env("LLAMA_ARG_OVERRIDE_TENSOR"));
    add_opt(common_arg(
        {"-cmoe", "--cpu-moe"},
        "keep all Mixture of Experts (MoE) weights in the CPU",
        [](common_params & params) {
            params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
        }
    ).set_env("LLAMA_ARG_CPU_MOE"));
    add_opt(common_arg(
        {"-ncmoe", "--n-cpu-moe"}, "N",
        "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            llm_add_n_cpu_ffn_overrides(value, LLM_FFN_EXPS_REGEX, params.tensor_buft_overrides);
        }
    ).set_env("LLAMA_ARG_N_CPU_MOE"));
    // ik_llama 894191d6e (#2644, MIT) common/common.cpp:2392-2406 (--moe-cache-mib in gpt_params_find_arg) and :3586
    // (usage line), ported to this fork's common_arg table. Same parse rule: a whole non-negative decimal MiB count
    // that still fits in size_t once multiplied by 1 MiB; anything else is rejected, not truncated.
    // RESEARCH-DONE: ik 894191d6e common/common.cpp:2392 -- 0 is upstream's own default (opt-in cache), not a switch-off.
    add_opt(common_arg(
        {"--moe-cache-mib"}, "N|auto",
        "total GPU cache budget in MiB for CPU-resident MoE experts (e.g. with -cmoe / -ncmoe), split across GPUs "
        "(default: 0 = no cache); 'auto' (or -1) lets --fit choose it from the free VRAM (one GPU, MoE model, --fit on) "
        "[EXPERIMENTAL]",
        [](common_params & params, const std::string & value) {
            // 'auto' = unslothai/llama.cpp PR 251 (common/arg.cpp @@ -2725,13 +2725,21 @@); the fit does the sizing (common/fit.cpp)
            if (value == "auto" || value == "-1") {
                params.moe_cache_auto = true;
                params.moe_cache_size = 0;
                return;
            }
            params.moe_cache_auto = false;
            uint64_t mib = 0;
            const char * first = value.c_str();
            const char * last  = first + value.size();
            const auto parsed = std::from_chars(first, last, mib);
            if (value.empty() || parsed.ec != std::errc() || parsed.ptr != last || value[0] == '-' ||
                    mib > std::numeric_limits<size_t>::max() / (1024*1024)) {
                throw std::invalid_argument(string_format(
                    "error: invalid value for --moe-cache-mib: %s (must be a non-negative MiB count)", value.c_str()));
            }
            params.moe_cache_size = (size_t) mib * 1024 * 1024;
        }
    ).set_env("LLAMA_ARG_MOE_CACHE_MIB"));
    // upstream llama.cpp e95426f16 (#30966) spelling and default; 'auto' is this fork's addition (src/llama-moe-cache.cpp calibrate_cpu_share)
    add_opt(common_arg(
        {"--moe-cache-cpu-share"}, "F|auto",
        string_format("share of the MoE cache misses that the CPU computes from the host experts instead of uploading them to the GPU once the cache is full, "
            "from 0 to 1, or 'auto' to measure the upload and CPU speeds at load (default: %.2f); needs --moe-cache-policy classlru (set for you if F > 0)",
            (double) params.moe_cache_cpu_share),
        [](common_params & params, const std::string & value) {
            if (value == "auto") {
                params.moe_cache_cpu_share = -1.0f;
                return;
            }
            const float share = std::stof(value);
            if (!(share >= 0.0f && share <= 1.0f)) {
                throw std::invalid_argument("invalid value for --moe-cache-cpu-share: " + value + " (0 to 1 or auto)");
            }
            params.moe_cache_cpu_share = share;
        }
    ).set_env("LLAMA_ARG_MOE_CACHE_CPU_SHARE"));
    add_opt(common_arg(
        {"--moe-cache-policy"}, "slru|classlru",
        "MoE cache replacement policy: slru = segmented LRU (default), classlru = LRFU/MQ-like frequency classes (upstream llama.cpp #31233)",
        [](common_params & params, const std::string & value) {
            if (value == "slru")          { params.moe_cache_policy = 0; }
            else if (value == "classlru") { params.moe_cache_policy = 1; }
            else { throw std::invalid_argument("invalid value for --moe-cache-policy: " + value + " (slru or classlru)"); }
        }
    ).set_env("LLAMA_ARG_MOE_CACHE_POLICY"));
    add_opt(common_arg(
        {"-ncffn", "--n-cpu-ffn"}, "N",
        "keep the dense FFN weights of the first N layers in the CPU\n"
        "(dense models; for MoE expert weights use --n-cpu-moe)",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            llm_add_n_cpu_ffn_overrides(value, LLM_FFN_DENSE_REGEX, params.tensor_buft_overrides);
        }
    ).set_env("LLAMA_ARG_N_CPU_FFN"));
    GGML_ASSERT(params.n_gpu_layers < 0); // string_format would need to be extended for a default >= 0
    add_opt(common_arg(
        {"-ngl", "--gpu-layers", "--n-gpu-layers"}, "N",
        string_format("max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)", params.n_gpu_layers == -1 ? "auto" : "all"),
        [](common_params & params, const std::string & value) {
            if (value == "auto") {
                params.n_gpu_layers = -1;
            } else if (value == "all") {
                params.n_gpu_layers = -2;
            } else {
                params.n_gpu_layers = std::stoi(value);
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: no usable GPU found, --gpu-layers option will be ignored\n");
                fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\n");
                fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
            }
        }
    ).set_env("LLAMA_ARG_N_GPU_LAYERS"));
    add_opt(common_arg(
        {"-sm", "--split-mode"}, "{none,layer,attn,graph,row,tensor}",
        "how to split the model across multiple GPUs, one of:\n"
        "- none: use one GPU only\n"
        "- layer (default): split layers and KV across GPUs (pipelined)\n"
        "- attn: split self-attention computations across GPUs (ik_llama)\n"
        "- graph: split computations across GPUs (ik_llama tensor-parallel graph)\n"
        "- row: split weight across GPUs by rows (parallelized)\n"
        "- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)",
        [](common_params & params, const std::string & value) {
            // attn / graph RESTORED 2026-09-27: the fork's pre-merge parser accepted them (common.cpp @ 35f0cac75:2111-2116)
            // and src/ consumes both modes at dozens of sites (llama-context.cpp:520, llama-load-tensors.cpp:280, ...),
            // but the merged upstream parser rejected them as "invalid value" -- the modes were unreachable.
            if (value == "none") {
                params.split_mode = LLAMA_SPLIT_MODE_NONE;
            } else if (value == "layer") {
                params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            } else if (value == "attn") {
                params.split_mode = LLAMA_SPLIT_MODE_ATTN;
            } else if (value == "graph") {
                params.split_mode = LLAMA_SPLIT_MODE_GRAPH;
            } else if (value == "row") {
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
            } else if (value == "tensor") {
                params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
            } else {
                throw std::invalid_argument("invalid value");
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the split mode has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_SPLIT_MODE"));
    add_opt(common_arg(
        {"-ts", "--tensor-split"}, "N0,N1,N2,...",
        "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1",
        [](common_params & params, const std::string & value) {
            std::string arg_next = value;

            // split string by , and /
            const std::regex regex{ R"([,/]+)" };
            std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
            std::vector<std::string> split_arg{ it, {} };
            if (split_arg.size() >= llama_max_devices()) {
                throw std::invalid_argument(
                    string_format("got %zu input configs, but system only has %zu devices", split_arg.size(), llama_max_devices())
                );
            }
            for (size_t i = 0; i < llama_max_devices(); ++i) {
                if (i < split_arg.size()) {
                    params.tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    params.tensor_split[i] = 0.0f;
                }
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting a tensor split has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_TENSOR_SPLIT"));
    add_opt(common_arg(
        {"-mg", "--main-gpu"}, "INDEX",
        string_format("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: %d)", params.main_gpu),
        [](common_params & params, int value) {
            params.main_gpu = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the main GPU has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_MAIN_GPU"));
    add_opt(common_arg(
        { "-fit", "--fit" }, "[on|off]",
        string_format("whether to adjust unset arguments to fit in device memory ('on' or 'off', default: '%s')", params.fit_params ? "on" : "off"),
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.fit_params = true;
            } else if (is_falsey(value)) {
                params.fit_params = false;
            } else {
                throw std::runtime_error(
                    string_format("error: unknown value for --fit: '%s'\n", value.c_str()));
            }
        }
    ).set_env("LLAMA_ARG_FIT"));
    add_opt(common_arg(
        { "-fitp", "--fit-print" }, "[on|off]",
        string_format("print the estimated required memory ('on' or 'off', default: '%s')", params.fit_params_print ? "on" : "off"),
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.fit_params_print = true;
            } else if (is_falsey(value)) {
                params.fit_params_print = false;
            } else {
                throw std::runtime_error(
                    string_format("error: unknown value for --fit-print: '%s'\n", value.c_str()));
            }
        }
    ).set_examples({LLAMA_EXAMPLE_FIT_PARAMS}).set_env("LLAMA_ARG_FIT_ESTIMATE"));
    add_opt(common_arg(
        { "-fitt", "--fit-target" }, "MiB0,MiB1,MiB2,...",
        string_format("target margin per device for --fit, comma-separated list of values, "
            "single value is broadcast across all devices, default: %zu", params.fit_params_target[0]/(1024*1024)),
        [](common_params & params, const std::string & value) {
            std::string arg_next = value;

            // split string by , and /
            const std::regex regex{ R"([,/]+)" };
            std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
            std::vector<std::string> split_arg{ it, {} };
            if (split_arg.size() >= llama_max_devices()) {
                throw std::invalid_argument(
                    string_format("got %zu input configs, but system only has %zu devices", split_arg.size(), llama_max_devices())
                );
            }
            if (split_arg.size() == 1) {
                std::fill(params.fit_params_target.begin(), params.fit_params_target.end(), std::stoull(split_arg[0]) * 1024*1024);
                return;
            }
            for (size_t i = 0; i < split_arg.size(); i++) {
                params.fit_params_target[i] = std::stoull(split_arg[i]) * 1024*1024;
            }
        }
    ).set_env("LLAMA_ARG_FIT_TARGET"));
    add_opt(common_arg(
        { "-fitc", "--fit-ctx" }, "N",
        string_format("minimum ctx size that can be set by --fit option, default: %" PRIu32, params.fit_params_min_ctx),
        [](common_params & params, int value) {
            params.fit_params_min_ctx = value;
        }
    ).set_env("LLAMA_ARG_FIT_CTX"));
    add_opt(common_arg(
        {"--model-checksum"}, "SHA256",
        "refuse to load the model unless the file's SHA-256 (64 hex characters) matches; a mismatch names the expected and the actual digest",
        [](common_params & params, const std::string & value) {
            params.model_checksum = value;
        }
    ).set_env("LLAMA_ARG_MODEL_CHECKSUM"));
    add_opt(common_arg(
        {"--check-tensors"},
        string_format("check model tensor data for invalid values (default: %s)", params.check_tensors ? "true" : "false"),
        [](common_params & params) {
            params.check_tensors = true;
        }
    ));
    add_opt(common_arg(
        {"-okv", "--override-kv"}, "KEY=TYPE:VALUE,...",   // -okv: fork short alias the upstream merge 279b4bfaf dropped
        "advanced option to override model metadata by key. to specify multiple overrides, either use comma-separated values.\n"
        "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false,tokenizer.ggml.add_eos_token=bool:false",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                if (!string_parse_kv_override(item.c_str(), params.kv_overrides)) {
                    throw std::runtime_error(string_format("error: Invalid type for KV override: %s\n", item.c_str()));
                }
            }
        }
    ));
    add_opt(common_arg(
        {"--op-offload"},
        {"--no-op-offload"},
        string_format("whether to offload host tensor operations to device (default: %s)", params.no_op_offload ? "false" : "true"),
        [](common_params & params, bool value) {
            params.no_op_offload = !value;
        }
    ));
    add_opt(common_arg(
        {"--lora"}, "FNAME",
        "path to LoRA adapter (use comma-separated values to load multiple adapters)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                params.lora_adapters.push_back({ item, 1.0, "", "", nullptr });
            }
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--lora-merge-strategy"}, "STRATEGY",
        "export-lora: how several adapters are combined: linear (default, add every delta with its scale), ties (prune each delta to --lora-merge-density by magnitude, elect a sign, "
        "merge the agreeing entries), dare_linear / dare_ties (random prune with rescale, then linear / ties). The per-adapter scale (--lora-scaled FNAME:SCALE) is the merge weight.",
        [](common_params & params, const std::string & value) {
            lora_merge_strategy s;
            if (!lora_merge_strategy_from_name(value, s)) {
                throw std::invalid_argument("--lora-merge-strategy: expected linear, ties, dare_linear or dare_ties, got '" + value + "'");
            }
            params.lora_merge_strategy = lora_merge_strategy_name(s);
        }
    ).set_examples({LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--lora-merge-density"}, "F",
        string_format("export-lora: fraction of each adapter delta kept by ties / dare_* (0..1, default: %.2f)", (double) params.lora_merge_density),
        [](common_params & params, const std::string & value) {
            const float d = std::stof(value);
            if (!(d >= 0.0f && d <= 1.0f)) {
                throw std::invalid_argument("--lora-merge-density: expected a number in [0, 1], got '" + value + "'");
            }
            params.lora_merge_density = d;
        }
    ).set_examples({LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--lora-merge-sign"}, "METHOD",
        "export-lora: how ties / dare_ties elect the majority sign: total (sign of the summed values, default) or frequency (sign of the summed signs)",
        [](common_params & params, const std::string & value) {
            lora_sign_method m;
            if (!lora_sign_method_from_name(value, m)) {
                throw std::invalid_argument("--lora-merge-sign: expected total or frequency, got '" + value + "'");
            }
            params.lora_merge_sign = m == LORA_SIGN_TOTAL ? "total" : "frequency";
        }
    ).set_examples({LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--lora-merge-seed"}, "N",
        "export-lora: random seed of the dare_* pruning (default: 0)",
        [](common_params & params, int value) {
            params.lora_merge_seed = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--lora-scaled"}, "FNAME:SCALE,...",
        "path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)\n"
        "note: use comma-separated values",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                auto parts = string_split<std::string>(item, ':');
                if (parts.size() != 2) {
                    throw std::invalid_argument("lora-scaled format: FNAME:SCALE");
                }
                params.lora_adapters.push_back({ parts[0], std::stof(parts[1]), "", "", nullptr });
            }
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--control-vector"}, "FNAME",
        "add a control vector\nnote: use comma-separated values to add multiple control vectors",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                params.control_vectors.push_back({ 1.0f, item, });
            }
        }
    ));
    add_opt(common_arg(
        {"--control-vector-scaled"}, "FNAME:SCALE,...",
        "add a control vector with user defined scaling SCALE\n"
        "note: use comma-separated values (format: FNAME:SCALE,...)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                auto parts = string_split<std::string>(item, ':');
                if (parts.size() != 2) {
                    throw std::invalid_argument("control-vector-scaled format: FNAME:SCALE");
                }
                params.control_vectors.push_back({ std::stof(parts[1]), parts[0] });
            }
        }
    ));
    add_opt(common_arg(
        {"--control-vector-layer-range"}, "START", "END",
        "layer range to apply the control vector(s) to, start and end inclusive",
        [](common_params & params, const std::string & start, const std::string & end) {
            params.control_vector_layer_start = std::stoi(start);
            params.control_vector_layer_end = std::stoi(end);
        }
    ));
    add_opt(common_arg(
        {"-a", "--alias"}, "STRING",
        "set model name aliases, comma-separated (to be used by API)",
        [](common_params & params, const std::string & value) {
            for (auto & alias : string_split<std::string>(value, ',')) {
                alias = string_strip(alias);
                if (!alias.empty()) {
                    params.model_alias.insert(alias);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ALIAS"));
    add_opt(common_arg(
        {"--tags"}, "STRING",
        "set model tags, comma-separated (informational, not used for routing)",
        [](common_params & params, const std::string & value) {
            for (auto & tag : string_split<std::string>(value, ',')) {
                tag = string_strip(tag);
                if (!tag.empty()) {
                    params.model_tags.insert(tag);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TAGS"));
    add_opt(common_arg(
        {"-m", "--model"}, "FNAME",
        ex == LLAMA_EXAMPLE_EXPORT_LORA
            ? "model path from which to load base model"
            : "model path to load",
        [](common_params & params, const std::string & value) {
            params.model.path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_MODEL"));
    add_opt(common_arg(
        {"-mu", "--model-url"}, "MODEL_URL",
        "model download url (default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.url = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_MODEL_URL"));
    add_opt(common_arg(
        { "-dr", "--docker-repo" }, "[<repo>/]<model>[:quant]",
        "Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest.\n"
        "example: gemma3\n"
        "(default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.docker_repo = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_DOCKER_REPO"));
    add_opt(common_arg(
        {"-hf", "-hfr", "--hf-repo"}, "<user>/<model>[:quant]",
        "Hugging Face model repository; quant is optional, case-insensitive, default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.\n"
        "mmproj is also downloaded automatically if available. to disable, add --no-mmproj\n"
        "example: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M\n"
        "(default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.hf_repo = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_HF_REPO"));
    add_opt(common_arg(
        {"-hff", "--hf-file"}, "FILE",
        "Hugging Face model file. If specified, it will override the quant in --hf-repo (default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.hf_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_HF_FILE"));
    add_opt(common_arg(
        {"--hf-revision"}, "REV",
        "Pin --hf-repo to a Hugging Face branch, tag or 40-hex commit (default: the main branch). A revision that names nothing is an error, "
        "never main. mmproj comes from the same revision. A pinned repo is not served from the local cache when the hub is unreachable "
        "(default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.hf_revision = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_HF_REVISION"));
    add_opt(common_arg(
        {"-hft", "--hf-token"}, "TOKEN",
        "Hugging Face access token (default: value from HF_TOKEN environment variable)",
        [](common_params & params, const std::string & value) {
            params.hf_token = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("HF_TOKEN"));
    add_opt(common_arg(
        {"--mtp"},
        "also download the multi-token prediction (MTP) head, if available (default: unused)",
        [](common_params & params) {
            spec_stages_push(params, COMMON_SPECULATIVE_TYPE_MTP);
        }
    ).set_examples({LLAMA_EXAMPLE_DOWNLOAD}));
    add_opt(common_arg(
        {"--dflash"},
        "also download the DFlash sidecar, if available (default: unused)",
        [](common_params & params) {
            spec_stages_push(params, COMMON_SPECULATIVE_TYPE_DFLASH);
        }
    ).set_examples({LLAMA_EXAMPLE_DOWNLOAD}));
    add_opt(common_arg(
        {"--eagle3"},
        "also download the Eagle3 sidecar, if available (default: unused)",
        [](common_params & params) {
            spec_stages_push(params, COMMON_SPECULATIVE_TYPE_EAGLE3);
        }
    ).set_examples({LLAMA_EXAMPLE_DOWNLOAD}));
    add_opt(common_arg(
        {"--context-file"}, "FNAME",
        "file to load context from (use comma-separated values to specify multiple files)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                std::ifstream file(item, std::ios::binary);
                if (!file) {
                    throw std::runtime_error(string_format("error: failed to open file '%s'\n", item.c_str()));
                }
                params.context_files.push_back(item);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--chunk-size"}, "N",
        string_format("minimum length of embedded text chunks (default: %d)", params.chunk_size),
        [](common_params & params, int value) {
            params.chunk_size = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--chunk-separator"}, "STRING",
        string_format("separator between chunks (default: '%s')", params.chunk_separator.c_str()),
        [](common_params & params, const std::string & value) {
            params.chunk_separator = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--junk"}, "N",
        string_format("number of times to repeat the junk text (default: %d)", params.n_junk),
        [](common_params & params, int value) {
            params.n_junk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"--pos"}, "N",
        string_format("position of the passkey in the junk text (default: %d)", params.i_pos),
        [](common_params & params, int value) {
            params.i_pos = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY}));
    add_opt(common_arg(
        {"-o", "--output", "--output-file"}, "FNAME",
        string_format("output file (default: '%s')", params.out_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.out_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_CVECTOR_GENERATOR, LLAMA_EXAMPLE_EXPORT_LORA, LLAMA_EXAMPLE_TTS, LLAMA_EXAMPLE_FINETUNE,
                    LLAMA_EXAMPLE_RESULTS, LLAMA_EXAMPLE_EXPORT_GRAPH_OPS, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--output-draft", "--draft-output", "--draft-output-file"}, "FNAME",
        // PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01:
        // SOURCE-READ: temp/_old_common.cpp (pre-merge common/common.cpp @ 35f0cac75:2943-2947,
        // usage row :3479) -- three aliases of one option writing params.out_file_draft
        // (common.h:1096, already declared, unreachable until this writer). imatrix only: the
        // paired draft-model imatrix output path, default derived from --output when unset (see
        // default_draft_imatrix_out_file() in tools/imatrix/imatrix.cpp).
        string_format("paired draft output file (default: derived from --output; current: '%s')",
                       params.out_file_draft.c_str()),
        [](common_params & params, const std::string & value) {
            params.out_file_draft = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"-ofreq", "--output-frequency"}, "N",
        string_format("output the imatrix every N iterations (default: %d)", params.n_out_freq),
        [](common_params & params, int value) {
            params.n_out_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--output-format"}, "{gguf,dat}",
        string_format("output format for imatrix file (default: %s)", params.imat_dat > 0 ? "dat" : "gguf"),
        [](common_params & params, const std::string & value) {
            /**/ if (value == "gguf") { params.imat_dat = -1; }
            else if (value == "dat")  { params.imat_dat = 1;  }
            else { throw std::invalid_argument("invalid output format"); }
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--save-frequency"}, "N",
        string_format("save an imatrix copy every N iterations (default: %d)", params.n_save_freq),
        [](common_params & params, int value) {
            params.n_save_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--process-output"},
        string_format("collect data for the output tensor (default: %s)", params.process_output ? "true" : "false"),
        [](common_params & params) {
            params.process_output = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--output-tensor-name"}, "NAME",
        // PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01:
        // SOURCE-READ: examples/imatrix/imatrix.cpp:270,332 (fork-only). The field already existed
        // unreachable on common_params (common.h:1088, default "output.weight") since the
        // 279b4bfaf restore pass added the struct member without a CLI writer -- this is the writer.
        string_format("name of the output tensor to collect imatrix data for with --process-output "
                       "or -lsim (default: '%s')", params.output_tensor_name.c_str()),
        [](common_params & params, const std::string & value) {
            params.output_tensor_name = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"-lsim", "--layer-similarity"},
        // SOURCE-READ: examples/imatrix/imatrix.cpp:228,1112-1113 (fork-only; parsed by imatrix's
        // own pre-common_params_parse loop there, not common/arg.cpp -- ported here as a proper
        // add_opt so tools/imatrix/imatrix.cpp can read it off common_params like every other flag).
        string_format("collect layer/attn/ffn cosine-similarity importance data (default: %s)",
                       params.collect_lsim ? "true" : "false"),
        [](common_params & params) {
            params.collect_lsim = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    // row S69-F1-PICK-IMATRIX-BLOCK-HESSIAN-OUTPUT-2026-10-06-01: optional block-diagonal X^T X sidecar for a GPTQ
    // quantizer. Format and accumulation: common/imatrix-hessian.h; collection: tools/imatrix/imatrix.cpp.
    add_opt(common_arg(
        {"--hessian-block"}, "N",
        string_format("also accumulate the N x N diagonal blocks of X^T X (sum over tokens of x x^T) for every matrix "
                      "and write them to a sidecar GGUF (see --hessian-out); memory K*N floats per matrix, e.g. 2 MiB "
                      "for K=4096, N=128; %d..%d, 0 = off (default: %d)",
                      IMATRIX_HESSIAN_BLOCK_MIN, IMATRIX_HESSIAN_BLOCK_MAX, params.imat_hessian_block),
        [](common_params & params, int value) {
            if (value != 0 && (value < IMATRIX_HESSIAN_BLOCK_MIN || value > IMATRIX_HESSIAN_BLOCK_MAX)) {
                throw std::invalid_argument(string_format("--hessian-block must be 0 (off) or %d..%d, got %d",
                                                          IMATRIX_HESSIAN_BLOCK_MIN, IMATRIX_HESSIAN_BLOCK_MAX, value));
            }
            params.imat_hessian_block = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--hessian-out"}, "FNAME",
        "with --hessian-block: sidecar output file (default: the imatrix output name with .gguf replaced by .hessian.gguf)",
        [](common_params & params, const std::string & value) {
            params.imat_hessian_out = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--hessian-max-mib"}, "N",
        string_format("with --hessian-block: refuse to run when the Hessian blocks of all matrices would need more than "
                      "N MiB of RAM (default: %d)", params.imat_hessian_max_mib),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("--hessian-max-mib must be >= 1");
            }
            params.imat_hessian_max_mib = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--nextn"},
        string_format("collect data for MTP/NextN layers (default: %s)", params.load_mtp ? "true" : "false"),
        [](common_params & params) {
            params.load_mtp = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--ppl"},
        {"--no-ppl"},
        string_format("whether to compute perplexity (default: %s)", params.compute_ppl ? "true" : "false"),
        [](common_params & params, bool value) {
            params.compute_ppl = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--chunk", "--from-chunk"}, "N",
        string_format("start processing the input from chunk N (default: %d)", params.i_chunk),
        [](common_params & params, int value) {
            params.i_chunk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--show-statistics"},
        string_format("show imatrix statistics and then exit (default: %s)", params.show_statistics ? "true" : "false"),
        [](common_params & params) {
            params.show_statistics = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--image-calib"}, "DIR",
        "calibrate the imatrix on the activations real images produce in the text model: every image in DIR is encoded with --mmproj and decoded as image tokens (needs -mm/--mmproj; may be combined with -f)",
        [](common_params & params, const std::string & value) {
            params.image_calib_dir = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--parse-special"},
        string_format("parse special tokens (chat, tool, etc) (default: %s)", params.parse_special ? "true" : "false"),
        [](common_params & params) {
            params.parse_special = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--ids"},
        string_format("only print the token IDs, in a Python-parseable list form like [1, 2, 3] (default: %s)", params.tokenize_ids ? "true" : "false"),
        [](common_params & params) {
            params.tokenize_ids = true;
        }
    ).set_examples({LLAMA_EXAMPLE_TOKENIZE}));
    add_opt(common_arg(
        {"--stdin"},
        string_format("read the prompt from stdin (takes precedence over -f/--file and -p/--prompt) (default: %s)", params.tokenize_stdin ? "true" : "false"),
        [](common_params & params) {
            params.tokenize_stdin = true;
        }
    ).set_examples({LLAMA_EXAMPLE_TOKENIZE}));
    add_opt(common_arg(
        {"--no-bos"},
        string_format("do not add a BOS token to the prompt, even if the model normally uses one (default: %s)", params.tokenize_no_bos ? "true" : "false"),
        [](common_params & params) {
            params.tokenize_no_bos = true;
        }
    ).set_examples({LLAMA_EXAMPLE_TOKENIZE}));
    add_opt(common_arg(
        {"--no-parse-special"},
        string_format("do not parse special tokens (chat, tool, etc) (default: %s)", !params.parse_special ? "true" : "false"),
        [](common_params & params) {
            params.parse_special = false;
        }
    ).set_examples({LLAMA_EXAMPLE_TOKENIZE}));
    add_opt(common_arg(
        {"--show-count"},
        string_format("print the total number of tokens (default: %s)", params.tokenize_show_count ? "true" : "false"),
        [](common_params & params) {
            params.tokenize_show_count = true;
        }
    ).set_examples({LLAMA_EXAMPLE_TOKENIZE}));
    add_opt(common_arg(
        {"-pps"},
        string_format("is the prompt shared across parallel sequences (default: %s)", params.is_pp_shared ? "true" : "false"),
        [](common_params & params) {
            params.is_pp_shared = true;
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-tgs"},
        string_format("is the text generation separated across the different sequences (default: %s)", params.is_tg_separate ? "true" : "false"),
        [](common_params & params) {
            params.is_tg_separate = true;
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-npp"}, "n0,n1,...",
        "number of prompt tokens",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pp.insert(params.n_pp.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"-ntg"}, "n0,n1,...",
        "number of text generation tokens",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_tg.insert(params.n_tg.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"-npl"}, "n0,n1,...",
        "number of parallel prompts",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pl.insert(params.n_pl.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"--embd-normalize"}, "N",
        string_format("normalisation for embeddings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)", params.embd_normalize),
        [](common_params & params, int value) {
            params.embd_normalize = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--embd-sort-by-length"},
        "pack the prompts into batches longest first (similar lengths share a batch, so fewer decode calls); results come back in the "
        "caller's order (default: as given)",
        [](common_params & params) {
            params.embd_sort_by_length = true;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--restrict-output-vocab"},
        "llama-server: when EVERY running request carries allowed_token_ids (and none reads raw probabilities: n_probs, logprobs, echo), multiply the output\n"
        "projection only by the union of those tokens instead of the whole vocabulary (CTranslate2's vocabulary map). Same tokens, same logits for them,\n"
        "less work per step; the saving is the output-weight share of a decode step (default: off)",
        [](common_params & params) {
            params.restrict_output_vocab = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--embd-output-format"}, "FORMAT",
        "empty = default, \"array\" = [[],[]...], \"json\" = openai style, \"json+\" = same \"json\" + cosine similarity matrix, \"raw\" = plain whitespace-delimited output (one embedding per line)",
        [](common_params & params, const std::string & value) {
            params.embd_out = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--embd-separator"}, "STRING",
        "separator of embeddings (default \\n) for example \"<#sep#>\"",
        [](common_params & params, const std::string & value) {
            params.embd_sep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--cls-separator"}, "STRING",
        "separator of classification sequences (default \\t) for example \"<#seq#>\"",
        [](common_params & params, const std::string & value) {
            params.cls_sep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--host"}, "HOST",
        string_format("IP addresses to listen on, comma-separated, or UNIX socket paths ending in .sock; with multiple TCP addresses, :: binds IPv6 only; overlapping addresses result in undefined behavior (default: %s)", params.hostnames[0].c_str()),
        [](common_params & params, const std::string & value) {
            params.hostnames.clear();
            for (auto & host : parse_csv_row(value)) {
                host = string_strip(host);
                if (!host.empty()) {
                    params.hostnames.push_back(host);
                }
            }
            if (params.hostnames.empty()) {
                throw std::invalid_argument("--host requires at least one address");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_HOST"));
    add_opt(common_arg(
        {"--port"}, "PORT",
        string_format("port to listen (default: %d)", params.port),
        [](common_params & params, int value) {
            params.port = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PORT"));
    add_opt(common_arg(
        {"--reuse-port"},
        string_format("allow multiple sockets to bind to the same port (default: %s)", params.reuse_port ? "enabled" : "disabled"),
        [](common_params & params) {
            params.reuse_port = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_REUSE_PORT"));
    add_opt(common_arg(
        {"--path"}, "PATH",
        string_format("path to serve static files from (default: %s)", params.public_path.c_str()),
        [](common_params & params, const std::string & value) {
            params.public_path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_STATIC_PATH"));
    add_opt(common_arg(
        {"--cors-origins"}, "ORIGINS",
        string_format(
            "comma-separated list of allowed origins for CORS (default: %s)\n"
            "if set to special value 'localhost', reflect the Origin header only if it is localhost",
        params.cors_origins.c_str()),
        [](common_params & params, const std::string & value) {
            params.cors_origins = value;
            params.cors_origins_explicit = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CORS_ORIGINS"));
    add_opt(common_arg(
        {"--allowed-hosts"}, "HOSTS",
        "comma-separated Host header names to accept when the server listens on loopback with no --api-key (default: loopback names and IP literals only)\n"
        "a web page that re-points its own domain at 127.0.0.1 (DNS rebinding) is refused; use this when a reverse proxy forwards another Host, or '*' to switch the check off",
        [](common_params & params, const std::string & value) {
            for (const auto & h : parse_csv_row(value)) {
                if (!h.empty()) {
                    params.allowed_hosts.push_back(h);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ALLOWED_HOSTS"));
    add_opt(common_arg(
        {"--cors-methods"}, "METHODS",
        string_format("comma-separated list of allowed methods for CORS (default: %s)", params.cors_methods.c_str()),
        [](common_params & params, const std::string & value) {
            params.cors_methods = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CORS_METHODS"));
    add_opt(common_arg(
        {"--cors-headers"}, "HEADERS",
        string_format("comma-separated list of allowed headers for CORS (default: %s)", params.cors_headers.c_str()),
        [](common_params & params, const std::string & value) {
            params.cors_headers = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CORS_HEADERS"));
    add_opt(common_arg(
        {"--cors-credentials"},
        {"--no-cors-credentials"},
        string_format(
            "whether to allow credentials for CORS (default: %s)\n"
            "note: if this is enabled and --cors-origins is set to * (default), the Origin header will be echoed back, and credentials will always be allowed",
        params.cors_credentials ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.cors_credentials = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CORS_CREDENTIALS"));
    add_opt(common_arg(
        {"--api-prefix"}, "PREFIX",
        string_format("prefix path the server serves from, without the trailing slash (default: %s)", params.api_prefix.c_str()),
        [](common_params & params, const std::string & value) {
            params.api_prefix = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_API_PREFIX"));
    add_opt(common_arg(
        {"--ui-config", "--webui-config"}, "JSON",
        "JSON that provides default UI settings (overrides UI defaults)",
        [](common_params & params, const std::string & value) {
            params.ui_config_json = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_UI_CONFIG"));
    add_opt(common_arg(
        {"--ui-config-file", "--webui-config-file"}, "PATH",
        "JSON file that provides default UI settings (overrides UI defaults)",
        [](common_params & params, const std::string & value) {
            params.ui_config_json = read_file(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_UI_CONFIG_FILE"));
    add_opt(common_arg(
        {"--ui-mcp-proxy", "--webui-mcp-proxy"},
        {"--no-ui-mcp-proxy", "--no-webui-mcp-proxy"},
        "experimental: whether to enable MCP CORS proxy - do not enable in untrusted environments (default: disabled)",
        [](common_params & params, bool value) {
            params.ui_mcp_proxy = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_UI_MCP_PROXY"));
    add_opt(common_arg(
        {"--tools"}, "TOOL1,TOOL2,...",
        "experimental: whether to enable built-in tools for AI agents - do not enable in untrusted environments (default: no tools)\n"
        "specify \"all\" to enable all tools\n"
        "available tools: read_file, file_glob_search, grep_search, exec_shell_command, write_file, edit_file, get_info\n"
        "note: for security reasons, this will limit --cors-origins to localhost by default",
        [](common_params & params, const std::string & value) {
            params.server_tools = parse_csv_row(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TOOLS"));
    add_opt(common_arg(
        {"--tools-runtime"}, "OPTION",
        "experimental: run tools in a separate runtime environment (default: none, use host environment)\n"
        "available options:\n"
        "  'docker:<image>', 'podman:<image>': spin up a new container and reuse it for all invocations, clean up on server exit\n"
        "  'docker-container:<id>', 'podman-container:<id>': use an existing container by ID, won't stop on server exit\n"
        "  'ssh:<target>': run tools on a remote POSIX host over SSH, key-based auth and a trusted host key are required\n",
        [](common_params & params, const std::string & value) {
            params.server_tools_runtime = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TOOLS_RUNTIME"));
    add_opt(common_arg(
        {"--tools-max-output"}, "N",
        "experimental: most bytes one tool call may return to the model, for read_file, exec_shell_command and get_info output and for the "
        "file_glob_search / grep_search listings; a cut is announced in the result "
        "(default: 0 = 1/8 of the context of one slot at 4 bytes a token, 16384 where no model is loaded)",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("error: --tools-max-output must be 0 or more");
            }
            params.server_tools_max_output = (size_t) value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TOOLS_MAX_OUTPUT"));
    add_opt(common_arg(
        {"--mcp-servers-config"}, "PATH",
        "experimental: path to JSON file with MCP server definitions (Cursor-compatible format) - do not enable in untrusted environments (default: none)\n"
        "note: for security reasons, this will limit --cors-origins to localhost by default",
        [](common_params & params, const std::string & value) {
            params.mcp_servers_config = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MCP_SERVERS_CONFIG"));
    add_opt(common_arg(
        {"--mcp-servers-json"}, "JSON",
        "experimental: inline JSON with MCP server definitions (Cursor-compatible format) - do not enable in untrusted environments (default: none)\n"
        "note: for security reasons, this will limit --cors-origins to localhost by default",
        [](common_params & params, const std::string & value) {
            params.mcp_servers_json = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MCP_SERVERS_JSON"));
    add_opt(common_arg(
        {"-ag", "--agent"},
        {"-no-ag", "--no-agent"},
        "whether to enable CORS proxy and all built-in tools - do not enable in untrusted environments (default: disabled)\n"
        "note: for security reasons, this will limit --cors-origins to localhost by default",
        [](common_params & params, bool value) {
            if (value) {
                params.server_tools = {"all"};
                params.ui_mcp_proxy = true;
            } else {
                params.server_tools.clear();
                params.ui_mcp_proxy = false;
            }
            // note: do not modify cors_origins here, as the options are not evaluated in order (user may explicitly set --cors-origins before --agent)
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_AGENT"));
    add_opt(common_arg(
        {"--ui", "--webui"},
        {"--no-ui", "--no-webui"},
        string_format("whether to enable the Web UI (default: %s)", params.ui ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.ui = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_UI"));
    add_opt(common_arg(
        {"--connect"},
        string_format("(experimental) open a peer-to-peer tunnel for server using llama-connect (default: %s)", params.server_connect ? "enabled" : "disabled"),
        [](common_params & params) {
            params.server_connect = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CONNECT"));
    add_opt(common_arg(
        {"--connect-code"}, "CODE",
        "manually specify a 40-character code for --connect (default: generate a new one for each run)",
        [](common_params & params, const std::string & value) {
            std::string val = value;
            string_replace_all(val, "-", "");
            string_replace_all(val, " ", "");
            if (val.size() != 40) {
                throw std::invalid_argument(string_format("error: invalid connect code '%s', must be 40 characters\n", value.c_str()));
            }
            params.server_connect_code = val;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CONNECT_CODE"));
    add_opt(common_arg(
        {"--embedding", "--embeddings"},
        string_format("restrict to only support embedding use case; use only with dedicated embedding models (default: %s)", params.embedding ? "enabled" : "disabled"),
        [](common_params & params) {
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_DEBUG}).set_env("LLAMA_ARG_EMBEDDINGS"));
    add_opt(common_arg(
        {"--rerank", "--reranking"},
        string_format("enable reranking endpoint on server (default: %s)", "disabled"),
        [](common_params & params) {
            params.embedding = true;
            params.pooling_type = LLAMA_POOLING_TYPE_RANK;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_RERANKING"));
    add_opt(common_arg(
        {"--api-key"}, "KEY",
        "API key to use for authentication, multiple keys can be provided as a comma-separated list (default: none)",
        [](common_params & params, const std::string & value) {
            for (const auto & key : parse_csv_row(value)) {
                if (!key.empty()) {
                    params.api_keys.push_back(key);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_API_KEY"));
    add_opt(common_arg(
        {"--api-key-file"}, "FNAME",
        "path to file containing API keys, one per line; lines starting with a hash are treated as comments; the file is re-read "
        "when it changes, so keys can be added or removed without a restart (default: none)",
        [](common_params & params, const std::string & value) {
            read_api_key_file(value, params.api_keys);
            params.api_key_files.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_API_KEY_FILE"));
    // SOURCE-READ: upstream/tabbyAPI/common/auth.py (AGPL-3.0, behaviour only, no code copied); nuances in server-auth.h
    add_opt(common_arg(
        {"--api-key-admin"}, "KEY",
        "admin API key (comma-separated list for several): opens the management routes (model load/unload, control vectors, LoRA, "
        "slot save/restore/erase, saved prompts, SQL session store, POST /props) in addition to everything a regular key opens. "
        "While at least one admin key is set, a regular key gets 403 on those routes; with none set every valid key opens every "
        "route, as before. Accepted as Authorization: Bearer, X-Api-Key or X-Admin-Key (default: none)",
        [](common_params & params, const std::string & value) {
            for (const auto & key : parse_csv_row(value)) {
                if (!key.empty()) {
                    params.api_keys_admin.push_back(key);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ADMIN_KEY"));
    add_opt(common_arg(
        {"--api-key-admin-file"}, "FNAME",
        "path to file containing admin API keys, one per line (# comment lines); re-read when it changes (default: none)",
        [](common_params & params, const std::string & value) {
            read_api_key_file(value, params.api_keys_admin);
            params.api_key_admin_files.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_API_KEY_ADMIN_FILE"));
    add_opt(common_arg(
        {"--ssl-key-file"}, "FNAME",
        "path to file a PEM-encoded SSL private key",
        [](common_params & params, const std::string & value) {
            params.ssl_file_key = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_KEY_FILE"));
    add_opt(common_arg(
        {"--ssl-cert-file"}, "FNAME",
        "path to file a PEM-encoded SSL certificate",
        [](common_params & params, const std::string & value) {
            params.ssl_file_cert = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_CERT_FILE"));
    add_opt(common_arg(
        {"--chat-template-kwargs"}, "STRING",
        "sets additional params for the json template parser, must be a valid json object string, e.g. '{\"key1\":\"value1\",\"key2\":\"value2\"}'",
        [](common_params & params, const std::string & value) {
            auto parsed = json::parse(value);
            for (const auto & item : parsed.items()) {
                if (item.key() == "enable_thinking") {
                    LOG_WRN("Setting 'enable_thinking' via --chat-template-kwargs is deprecated. "
                            "Use --reasoning on / --reasoning off instead.\n");
                }
                if (item.key() == "preserve_reasoning") {
                    LOG_WRN("Setting 'preserve_reasoning' via --chat-template-kwargs is deprecated. "
                            "Use --reasoning-preserve / --no-reasoning-preserve instead.\n");
                }
                params.default_template_kwargs[item.key()] = item.value().dump();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_CHAT_TEMPLATE_KWARGS"));
    add_opt(common_arg(
        {"--chat-template-kwargs-force"}, "STRING",
        "like --chat-template-kwargs, but these keys REPLACE the same keys of a request's chat_template_kwargs (and win over "
        "its reasoning_effort for a forced reasoning_effort/enable_thinking), must be a valid json object string",
        [](common_params & params, const std::string & value) {
            auto parsed = json::parse(value);
            for (const auto & item : parsed.items()) {
                params.force_template_kwargs[item.key()] = item.value().dump();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CHAT_TEMPLATE_KWARGS_FORCE"));
    add_opt(common_arg(
        {"-to", "--timeout"}, "N",
        string_format("server read/write timeout in seconds (default: %d)", params.timeout_read),
        [](common_params & params, int value) {
            params.timeout_read  = value;
            params.timeout_write = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TIMEOUT"));
    add_opt(common_arg(
        {"--sse-ping-interval"}, "N",
        string_format("server SSE ping interval in seconds (-1 = disabled, default: %d)", params.sse_ping_interval),
        [](common_params & params, int value) {
            params.sse_ping_interval = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSE_PING_INTERVAL"));
    add_opt(common_arg(
        {"--max-queued-requests"}, "N",
        string_format("admission ceiling: completion requests allowed in flight (queued or running) at once; a request "
                      "beyond it is answered 503 at once instead of queueing (default: %d, 0 = no limit)", params.max_queued_requests),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value: cannot be negative");
            }
            params.max_queued_requests = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MAX_QUEUED_REQUESTS"));
    add_opt(common_arg(
        {"--max-queued-tokens"}, "N",
        string_format("admission ceiling on the prompt tokens of requests still in prefill, a time-to-first-token guard "
                      "(about target_TTFT * prefill tokens/s); a request that would exceed it is answered 503, except when "
                      "nothing else is in prefill (default: %lld, 0 = no limit)", (long long) params.max_queued_tokens),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value: cannot be negative");
            }
            params.max_queued_tokens = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MAX_QUEUED_TOKENS"));
    // SOURCE-READ: mlflow/mlflow@7d2713204b mlflow/gateway/config.py:493-497,560-572 (Limit{calls, renewal_period}) and
    // app.py:277-279 (keyed by remote address), Apache-2.0; behaviour, own code (tools/server/server-ratelimit.h)
    add_opt(common_arg(
        {"--rate-limit"}, "CALLS/PERIOD",
        "per-client request rate limit: at most CALLS POST requests per PERIOD from one client (see --rate-limit-key); the "
        "next ones are answered 429 with Retry-After = the time left in the client's window, and other clients are not "
        "affected. PERIOD is a duration (1s, 500ms, 1m30s, bare number = seconds) or a unit (second, minute, hour, day, "
        "\"10 minutes\"). Fixed window, starting at the client's first request. GET/OPTIONS and /health are not counted "
        "(default: none = no limit)",
        [](common_params & params, const std::string & value) {
            int64_t calls = 0, period_ms = 0;
            const bool ok = server_rate_limit::parse(value, calls, period_ms, [](const std::string & s, int64_t & ms) {
                return string_parse_duration_ms(s, ms, /* bare_number_is_seconds */ true);
            });
            if (!ok) {
                throw std::invalid_argument("invalid --rate-limit value (want CALLS/PERIOD, e.g. 60/minute or 2/1s): " + value);
            }
            params.rate_limit_calls     = calls;
            params.rate_limit_period_ms = period_ms;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_RATE_LIMIT"));
    add_opt(common_arg(
        {"--rate-limit-key"}, "{addr,key}",
        "what --rate-limit counts per: 'addr' = the remote address (default), 'key' = the API key the request presents "
        "(Authorization: Bearer / X-Api-Key / X-Admin-Key; a request without one is counted by address). Behind a reverse "
        "proxy every request has the proxy's address, so use 'key' there",
        [](common_params & params, const std::string & value) {
            if (value != "addr" && value != "key") {
                throw std::invalid_argument("invalid --rate-limit-key value (want addr or key): " + value);
            }
            params.rate_limit_key = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_RATE_LIMIT_KEY"));
    // SOURCE-READ: LostRuins/koboldcpp@36688de9 koboldcpp.py:7194-7204 (--maxrequestsize, MB, checked against
    // Content-Length before the body is read) and :7205-7224 (chunked bodies capped too), AGPL-3.0: behaviour only,
    // no code copied. The cap itself is cpp-httplib's payload limit (413 for both framings, reading stops at the cap);
    // this exposes it and gives the 413 a JSON body (tools/server/server-http.cpp).
    add_opt(common_arg(
        {"--max-request-size"}, "MiB",
        string_format("largest HTTP request body accepted, in MiB; a larger one (Content-Length or chunked framing) is "
                      "refused with HTTP 413 and a JSON error, and nothing past the cap is read (default: %d, 0 = no limit)",
                      params.max_request_size_mib),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid --max-request-size value: cannot be negative");
            }
            params.max_request_size_mib = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MAX_REQUEST_SIZE"));
    add_opt(common_arg(
        {"--threads-http"}, "N",
        string_format("number of threads used to process HTTP requests (default: %d)", params.n_threads_http),
        [](common_params & params, int value) {
            params.n_threads_http = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_THREADS_HTTP"));
    add_opt(common_arg(
        {"--cache-prompt"},
        {"--no-cache-prompt"},
        string_format("whether to enable prompt caching (default: %s)", params.cache_prompt ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.cache_prompt = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CACHE_PROMPT"));
    add_opt(common_arg(
        {"--cache-reuse"}, "N",
        string_format(
            "min chunk size to attempt reusing from the cache via KV shifting, requires prompt caching to be enabled (default: %d)\n"
            "[(card)](https://ggml.ai/f0.png)", params.n_cache_reuse
        ),
        [](common_params & params, int value) {
            params.n_cache_reuse = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CACHE_REUSE"));
    add_opt(common_arg(
        {"--metrics"},
        string_format("enable prometheus compatible metrics endpoint (default: %s)", params.endpoint_metrics ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_metrics = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_METRICS"));
    // RESEARCH-DONE: ik_llama.cpp 55813a5d7 common/common.cpp (--memory); CAPABILITY-GENUINELY-NA: adds a read-only endpoint, removes nothing
    add_opt(common_arg(
        {"--memory"},
        string_format("enable the /memory endpoint reporting per-device memory use (default: %s)", params.endpoint_memory ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_memory = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_MEMORY"));
    add_opt(common_arg(
        {"--props"},
        string_format("enable changing global properties via POST /props (default: %s)", params.endpoint_props ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_props = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_PROPS"));
    add_opt(common_arg(
        {"--slots"},
        {"--no-slots"},
        string_format("expose slots monitoring endpoint (default: %s)", params.endpoint_slots ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.endpoint_slots = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_SLOTS"));
    add_opt(common_arg(
        {"--slot-save-path"}, "PATH",
        "path to save slot kv cache (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.slot_save_path = value;
            if (!fs_is_directory(params.slot_save_path)) {
                throw std::invalid_argument("not a directory: " + value);
            }
            // if doesn't end with DIRECTORY_SEPARATOR, add it
            if (!params.slot_save_path.empty() && params.slot_save_path[params.slot_save_path.size() - 1] != DIRECTORY_SEPARATOR) {
                params.slot_save_path += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SLOT_SAVE_PATH"));
    add_opt(common_arg(
        {"--sql-save-file"}, "PATH",
        "path to a SQLite database file for the server-side session key/value store "
        "(POST/GET /load,/save,/rename,/all,/sessions,/delete,/vacuum; default: disabled). "
        "Created if it does not exist. Only has an effect on a server built with "
        "-DLLAMA_SERVER_SQLITE3=ON; ported from the pre-merge fork's examples/server, row "
        "S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G3.",
        [](common_params & params, const std::string & value) {
            params.sql_save_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--sqlite-zstd-ext-file"}, "PATH",
        "path to the sqlite-zstd loadable extension (github.com/phiresky/sqlite-zstd, LGPL-3.0, loaded at run "
        "time -- never linked or vendored) for the --sql-save-file store. When it loads, the transparent-"
        "compression routes /zstd_get_configs, /zstd_incremental_maintenance, /zstd_enable_transparent and "
        "/zstd_update_transparent become available (default: not loaded). Needs --sql-save-file and a server "
        "built with -DLLAMA_SERVER_SQLITE3=ON. Group G4 of row "
        "S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01.",
        [](common_params & params, const std::string & value) {
            params.sqlite_zstd_ext_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--media-path"}, "PATH",
        "directory for loading local media files; files can be accessed via file:// URLs using relative paths (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.media_path = value;
            if (!fs_is_directory(params.media_path)) {
                throw std::invalid_argument("not a directory: " + value);
            }
            // if doesn't end with DIRECTORY_SEPARATOR, add it
            if (!params.media_path.empty() && params.media_path[params.media_path.size() - 1] != DIRECTORY_SEPARATOR) {
                params.media_path += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--models-dir"}, "PATH",
        "directory containing models for the router server (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.models_dir = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_DIR"));
    add_opt(common_arg(
        {"--models-preset"}, "PATH",
        "path to INI file containing model presets for the router server (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.models_preset = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_PRESET"));
    add_opt(common_arg(
        {"--models-max"}, "N",
        string_format("for router server, maximum number of models to load simultaneously (default: %d, 0 = unlimited)", params.models_max),
        [](common_params & params, int value) {
            params.models_max = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_MAX"));
    add_opt(common_arg(
        {"--models-snapshot"}, "FILE",
        "for router server, keep the set of loaded models in FILE: rewritten (atomically) after every load and unload, "
        "and loaded again at the next start, after the load-on-startup presets and within --models-max. A preset that "
        "sets load-on-startup for a model (to either value) decides that model, not the snapshot (default: none = off)",
        [](common_params & params, const std::string & value) {
            params.models_snapshot = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_SNAPSHOT"));
    add_opt(common_arg(
        {"--reload-allow-empty"},
        "for router server, let a preset reload that finds no models unload every running model "
        "(default: refuse and keep the current set)",
        [](common_params & params) {
            params.models_reload_allow_empty = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_RELOAD_ALLOW_EMPTY"));
    add_opt(common_arg(
        {"--models-vram-admit"}, "F",
        string_format("for router server, VRAM-aware admission as in ollama's scheduler: before loading a model, "
            "evict idle models (least recently used first) until its predicted memory fits in F x free device "
            "memory, read through NVML; the prediction is the model's measured footprint from its last load, "
            "else its GGUF size (default: %.2f, 0 = off; ollama uses 0.8)", (double) params.models_vram_admit),
        [](common_params & params, const std::string & value) {
            params.models_vram_admit = std::stof(value);
            if (params.models_vram_admit < 0.0f || params.models_vram_admit > 1.0f) {
                throw std::invalid_argument("--models-vram-admit must be in [0, 1]");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_VRAM_ADMIT"));
    // CAPABILITY-GENUINELY-NA: router memory accounting option text, not a model capability
    add_opt(common_arg(
        {"--models-memory-margin"}, "N",
        string_format("for router server, MiB of device memory to keep free, unload LRU models until a new one fits "
            "(free device memory is the sum over devices, read through NVML; same admission as --models-vram-admit, "
            "both must hold when both are set) (default: %d, 0 = off)", params.models_memory_margin),
        [](common_params & params, int value) {
            params.models_memory_margin = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_MEMORY_MARGIN"));
    add_opt(common_arg(
        {"--models-autoload"},
        {"--no-models-autoload"},
        string_format("for router server, whether to automatically load models (default: %s)", params.models_autoload ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.models_autoload = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_AUTOLOAD"));
    // after TorchServe's model-API switch (pytorch/serve@62c4d6a ManagementRequestHandler.java:80-83, 101: register and
    // unregister only when the model API is enabled; list and describe stay open). Apache-2.0, idea only, no code copied.
    // Default ON, so nothing changes unless an operator opts out.
    // DISABLE-REVERSIBLE: --models-register (default on); row S69-TORCHSERVE-PICK-MODELS-REGISTER-GATE-FLAG-2026-10-15-01
    add_opt(common_arg(
        {"--models-register"},
        {"--no-models-register"},
        string_format("for router server, whether clients may register (POST /models, downloads from HF) and remove "
                      "(DELETE /models) models at runtime; load, unload and list are not affected (default: %s)",
                      params.models_register ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.models_register = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_REGISTER"));
    // after TorchServe's allowed_urls (pytorch/serve@62c4d6a1 archive/utils/ArchiveUtils.java:75-90, docs/configuration.md:314-316):
    // runtime registration may download only from repos an operator named. One pattern per use (a regex may hold a comma, so no
    // comma list). Deliberately no environment name: TorchServe reads its list from the environment only behind a second switch,
    // so that an environment variable cannot widen it. Apache-2.0, idea only, no code copied.
    // row S69-TORCHSERVE-PICK-MODELS-REGISTER-ALLOWLIST-2026-10-18-01
    add_opt(common_arg(
        {"--models-register-allow"}, "REGEX",
        "for router server, a Hugging Face repo that POST /models may register and download, as a regex that must match the "
        "WHOLE name case-insensitively (e.g. 'bartowski/.*'); repeat the flag for several. Unset: any repo (default). "
        "A name containing \"..\" is always refused",
        [](common_params & params, const std::string & value) {
            if (value.empty()) {
                throw std::invalid_argument("--models-register-allow: empty pattern");
            }
            try {
                std::regex re(value, std::regex::ECMAScript | std::regex::icase);
                (void) re;
            } catch (const std::regex_error & e) {
                throw std::invalid_argument("--models-register-allow: invalid regex '" + value + "': " + e.what());
            }
            params.models_register_allow.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--modelfile"}, "FILE",
        "Ollama Modelfile (FROM is informational). Its PARAMETER lines become defaults for /api/chat and /api/generate (a request's options{} wins), "
        "SYSTEM is used when the request has none, MESSAGE lines are prepended to /api/chat history; /api/show reports it. "
        "TEMPLATE (a Go template) is shown but not applied: the model's own Jinja template is used",
        [](common_params & params, const std::string & value) {
            params.ollama_modelfile = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELFILE"));
    add_opt(common_arg(
        {"--ollama-model"}, "NAME",
        "serve a model from an Ollama model store: NAME (e.g. llama3.2:3b, user/model:tag) is resolved through the store's manifest to its GGUF blob, "
        "projector (--mmproj) and LoRA adapter layers; the manifest's system / params / messages / license layers seed the Ollama API like --modelfile",
        [](common_params & params, const std::string & value) {
            params.ollama_model = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_OLLAMA_MODEL"));
    add_opt(common_arg(
        {"--ollama-models-dir"}, "DIR",
        "Ollama model store used by --ollama-model and by /api/copy and /api/delete (default: $OLLAMA_MODELS, else ~/.ollama/models)",
        [](common_params & params, const std::string & value) {
            params.ollama_models_dir = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_OLLAMA_MODELS_DIR"));
    add_opt(common_arg(
        {"--ollama-store-write"},
        "allow the Ollama API's /api/copy and /api/delete to modify the model store (--ollama-models-dir); default: read-only (they answer 403)",
        [](common_params & params) {
            params.ollama_store_write = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_OLLAMA_STORE_WRITE"));
    add_opt(common_arg(
        {"--ollama-env"},
        "take defaults from Ollama's environment: OLLAMA_KV_CACHE_TYPE (-ctk/-ctv), OLLAMA_FLASH_ATTENTION (-fa), OLLAMA_CONTEXT_LENGTH (-c), "
        "OLLAMA_NUM_PARALLEL (-np), OLLAMA_KEEP_ALIVE (--models-keep-alive), OLLAMA_MAX_LOADED_MODELS (--models-max); an explicit fork flag wins",
        [](common_params & params) {
            params.ollama_env = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_OLLAMA_ENV"));
    add_opt(common_arg(
        {"--ollama-go-template"},
        "with --ollama-model: when the GGUF carries no chat template, translate the store's Go TEMPLATE layer to Jinja and use it (a subset of Go template syntax; "
        "an unsupported construct is refused with a message and the model keeps having no template)",
        [](common_params & params) {
            params.ollama_go_template = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_OLLAMA_GO_TEMPLATE"));
    add_opt(common_arg(
        {"--ollama-strict-model-names"},
        "on /api/chat, /api/generate, /api/embed and /api/show answer 404 \"model 'x' not found\" unless the requested name resolves to the loaded model "
        "after Ollama name normalisation (host/namespace/tag defaults, case-insensitive); default: accept any name",
        [](common_params & params) {
            params.ollama_strict_names = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_OLLAMA_STRICT_MODEL_NAMES"));
    add_opt(common_arg(
        {"--models-keep-alive"}, "DURATION",
        "for router server, how long a model stays loaded after its last request before it is unloaded and its process exits, "
        "releasing its memory and GPU context (default: -1 = never); seconds or a duration like 5m, 1h30m; "
        "0 = unload right after each request. A request's own \"keep_alive\" field overrides it, as in ollama",
        [](common_params & params, const std::string & value) {
            if (!string_parse_duration_ms(value, params.models_keep_alive_ms, /* bare_number_is_seconds */ true)) {
                throw std::invalid_argument("invalid --models-keep-alive value: " + value);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_KEEP_ALIVE"));
    // RESEARCH-DONE: mudler/LocalAI pkg/model/loader.go:113-216 and core/cli/run.go:111 read (MIT); behaviour, own code
    add_opt(common_arg(
        {"--model-load-cooldown"}, "DURATION",
        "for router server, after a model fails to load, answer requests for it with 503 + Retry-After for this long instead of "
        "respawning a loader on every poll; doubles per consecutive failure up to --model-load-cooldown-max, a successful load "
        "resets it, POST /models/load bypasses it (default: 10s); seconds or a duration like 30s, 2m; 0 = feature not in use",
        [](common_params & params, const std::string & value) {
            if (!string_parse_duration_ms(value, params.models_load_cooldown_ms, /* bare_number_is_seconds */ true) ||
                params.models_load_cooldown_ms < 0) {
                throw std::invalid_argument("invalid --model-load-cooldown value: " + value);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODEL_LOAD_COOLDOWN"));
    add_opt(common_arg(
        {"--model-load-cooldown-max"}, "DURATION",
        "for router server, ceiling of the --model-load-cooldown doubling (default: 5m)",
        [](common_params & params, const std::string & value) {
            if (!string_parse_duration_ms(value, params.models_load_cooldown_max_ms, /* bare_number_is_seconds */ true) ||
                params.models_load_cooldown_max_ms <= 0) {
                throw std::invalid_argument("invalid --model-load-cooldown-max value: " + value);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODEL_LOAD_COOLDOWN_MAX"));
    add_opt(common_arg(
        {"--jinja"},
        {"--no-jinja"},
        string_format("whether to use jinja template engine for chat (default: %s)", params.use_jinja ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.use_jinja = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_MTMD}).set_env("LLAMA_ARG_JINJA"));
    add_opt(common_arg(
        {"--reasoning-format"}, "FORMAT",
        "controls whether thought tags are allowed and/or extracted from the response, and in which format they're returned; one of:\n"
        "- none: leaves thoughts unparsed in `message.content`\n"
        "- deepseek: puts thoughts in `message.reasoning_content`\n"
        "- deepseek-legacy: keeps `<think>` tags in `message.content` while also populating `message.reasoning_content`\n"
        "(default: auto)",
        [](common_params & params, const std::string & value) {
            params.reasoning_format = common_reasoning_format_from_name(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_THINK"));
    add_opt(common_arg(
        {"-rea", "--reasoning"}, "[on|off|auto]",
        "Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template))",
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.enable_reasoning = 1;
                params.default_template_kwargs["enable_thinking"] = "true";
            } else if (is_falsey(value)) {
                params.enable_reasoning = 0;
                params.default_template_kwargs["enable_thinking"] = "false";
            } else if (is_autoy(value)) {
                params.enable_reasoning = -1;
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --reasoning: '%s'\n", value.c_str()));
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_REASONING"));
    add_opt(common_arg(
        {"--reasoning-effort"}, "LEVEL",
        "reasoning effort level given to the chat template: 'default' to keep the template default,\n"
        "or a level such as 'minimal', 'low', 'medium', 'high', 'xhigh' or 'max' (default: default)",
        [](common_params & params, const std::string & value) {
            if (value == "default") {
                params.default_template_kwargs.erase("reasoning_effort");
            } else {
                params.default_template_kwargs["reasoning_effort"] = json(value).dump();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_REASONING_EFFORT"));
    add_opt(common_arg(
        {"--reasoning-budget"}, "N",
        "token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)",
        [](common_params & params, int value) {
            if (value < -1) { throw std::invalid_argument("invalid value"); }
            params.sampling.reasoning_budget_tokens = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_THINK_BUDGET"));
    add_opt(common_arg(
        {"--reasoning-budget-message"}, "MESSAGE",
        "message injected before the end-of-thinking tag when reasoning budget is exhausted (default: none)",
        [](common_params & params, const std::string & value) {
            params.sampling.reasoning_budget_message = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_THINK_BUDGET_MESSAGE"));
    add_opt(common_arg(
        {"--reasoning-preserve"},
        {"--no-reasoning-preserve"},
        "preserve reasoning trace in the full history, not just the last assistant message (default: enabled)\n"
        "compatible with certain templates having 'supports_preserve_reasoning' capability\n"
        "example: https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking",
        [](common_params & params, bool value) {
            if (value) {
                params.default_template_kwargs["preserve_reasoning"] = "true";
            } else {
                params.default_template_kwargs["preserve_reasoning"] = "false";
            }
            params.preserve_reasoning_specified = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_REASONING_PRESERVE"));
    add_opt(common_arg(
        // the fork's spelling (pre-merge common.cpp @ 35f0cac75:572, field think_tokens -- old
        // server-context.cpp:353-358,1022-1090 @ 35f0cac75) -- row S69-FORK-SERVER-FEATURES-
        // STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, item 2. Controls whether a served
        // model's own <think>...</think> reasoning-trace tokens are excluded when matching prompts
        // against cached slots / the RAM prompt cache -- unrelated to --reasoning-format/-budget,
        // which shape what the CLIENT sees, not how the SERVER decides prompt-cache reuse.
        {"--reasoning-tokens"}, "{none,auto,BEGIN,END}",
        "exclude reasoning-trace tokens when matching a prompt against cached slots / the RAM prompt "
        "cache, so turns whose visible prefix matches don't miss the cache just because a differing "
        "past reasoning trace is still attached: none = compare full token streams (no exclusion), "
        "auto = exclude <think>...</think> spans (default), or pass \"BEGIN,END\" for custom "
        "delimiters (default: auto)",
        [](common_params & params, const std::string & value) {
            params.think_tokens = thinking_tokens_from_string(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--chat-template"}, "JINJA_TEMPLATE",
        string_format(
            "set custom jinja chat template (default: template taken from model's metadata)\n"
            "if suffix/prefix are specified, template will be disabled\n"
            "only commonly used templates are accepted (unless --jinja is set before this flag):\n"
            "list of built-in templates:\n%s", list_builtin_chat_templates().c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.chat_template = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_MTMD}).set_env("LLAMA_ARG_CHAT_TEMPLATE"));
    add_opt(common_arg(
        {"--chat-template-file"}, "JINJA_TEMPLATE_FILE",
        string_format(
            "set custom jinja chat template file (default: template taken from model's metadata)\n"
            "if suffix/prefix are specified, template will be disabled\n"
            "only commonly used templates are accepted (unless --jinja is set before this flag):\n"
            "list of built-in templates:\n%s", list_builtin_chat_templates().c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.chat_template = read_file(value);
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CHAT_TEMPLATE_FILE"));
    add_opt(common_arg(
        {"--skip-chat-parsing"},
        {"--no-skip-chat-parsing"},
        string_format(
            "force a pure content parser, even if a Jinja template is specified; model will output everything "
            "in the content section, including any reasoning and/or tool calls (default: disabled)"
        ),
        [](common_params & params, bool value) {
            params.force_pure_content_parser = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SKIP_CHAT_PARSING"));
    add_opt(common_arg(
        {"--prefill-assistant"},
        {"--no-prefill-assistant"},
        string_format(
            "whether to prefill the assistant's response if the last message is an assistant message (default: prefill enabled)\n"
            "when this flag is set, if the last message is an assistant message then it will be treated as a full message and not prefilled\n"
        ),
        [](common_params & params, bool value) {
            params.prefill_assistant = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PREFILL_ASSISTANT"));
    add_opt(common_arg(
        {"--tool-arg-coercion"},
        {"--no-tool-arg-coercion"},
        string_format(
            "repair the argument types of parsed tool calls to the types the request's tool schemas declare "
            "(\"5\" -> 5 for an integer, \"true\" -> true for a boolean, a JSON-encoded string -> array/object), "
            "so strict clients accept calls from small or quantised models; applies to non-streaming responses "
            "(a request field \"tool_arg_coercion\" overrides it) (default: %s)",
            params.tool_arg_coercion ? "enabled" : "disabled"
        ),
        [](common_params & params, bool value) {
            params.tool_arg_coercion = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TOOL_ARG_COERCION"));
    add_opt(common_arg(
        {"--tool-call-format-json"}, "JSON|@FILE",
        "tell the server how this model writes XML-style tool calls instead of relying on the chat template: a JSON object with "
        "the markers scope_start, tool_start, tool_sep, key_start, key_val_sep, key_val_sep2, val_end, tool_end, scope_end and "
        "raw_argval, e.g. '{\"tool_start\":\"<call name=\\\"\",\"tool_sep\":\"\\\">\",\"key_start\":\"<p k=\\\"\","
        "\"key_val_sep\":\"\\\">\",\"val_end\":\"</p>\",\"tool_end\":\"</call>\"}' (a leading @ reads the JSON from a file). "
        "Replaces the template's own tool-call parser and grammar for requests that offer tools; the JSON is validated at startup "
        "(default: none)",
        [](common_params & params, const std::string & value) {
            const std::string text = (!value.empty() && value[0] == '@') ? read_file(value.substr(1)) : value;
            common_chat_xml_tool_format_parse(text); // throws std::invalid_argument naming the field
            params.tool_call_format = text;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TOOL_CALL_FORMAT_JSON"));
    add_opt(common_arg(
        {"--tool-call-format-preset"}, "NAME",
        "same as --tool-call-format-json with a shipped format: qwen3-coder, minimax-m2, glm-4.5 or seed-oss "
        "(default: none)",
        [](common_params & params, const std::string & value) {
            const std::string text = common_chat_xml_tool_format_preset_json(value); // throws on an unknown name
            common_chat_xml_tool_format_parse(text);
            params.tool_call_format = text;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TOOL_CALL_FORMAT_PRESET"));
    add_opt(common_arg(
        {"-sps", "--slot-prompt-similarity"}, "SIMILARITY",
        string_format("how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", params.slot_prompt_similarity),
        [](common_params & params, const std::string & value) {
            params.slot_prompt_similarity = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--slot-evict-lookahead"}, "N",
        string_format("when a request must evict an idle slot (least recently used), first look at the next N waiting requests "
                      "and keep a slot one of them would reuse by prompt similarity (--slot-prompt-similarity); falls back to "
                      "plain LRU when every idle slot is wanted (default: %d, 0 = plain LRU)", params.slot_evict_lookahead),
        [](common_params & params, int value) {
            params.slot_evict_lookahead = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SLOT_EVICT_LOOKAHEAD"));
    add_opt(common_arg(
        {"--lora-init-without-apply"},
        string_format("load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", params.lora_init_without_apply ? "enabled" : "disabled"),
        [](common_params & params) {
            params.lora_init_without_apply = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--sleep-idle-seconds"}, "SECONDS",
        string_format("number of seconds of idleness after which the server will sleep (default: %d; -1 = disabled)", params.sleep_idle_seconds),
        [](common_params & params, int value) {
            if (value == 0 || value < -1) {
                throw std::invalid_argument("invalid value: cannot be 0 or less than -1");
            }
            params.sleep_idle_seconds = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--forward-watchdog-soft-timeout"}, "SECONDS",
        string_format("log a stall when no decode step finished for SECONDS while a batch is in flight, once per stall (default: %d; 0 = off)", params.forward_watchdog_soft_s),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value: cannot be negative");
            }
            params.forward_watchdog_soft_s = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--forward-watchdog-timeout"}, "SECONDS",
        string_format("end the process SECONDS after the last decode step finished while a batch is in flight, after a 5 s grace, so a supervisor restarts it (default: %d; 0 = off)", params.forward_watchdog_hard_s),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value: cannot be negative");
            }
            params.forward_watchdog_hard_s = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--kv-audit-interval"}, "N",
        string_format("debug: every N decode steps check each slot's token/position canary chain, the KV cell count against slot tokens, and the sampled logits for NaN/+inf; violations are logged, never fatal (default: %d; 0 = off)", params.kv_audit_interval),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value: cannot be negative");
            }
            params.kv_audit_interval = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--simple-io"},
        "use basic IO for better compatibility in subprocesses and limited consoles",
        [](common_params & params) {
            params.simple_io = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--positive-file"}, "FNAME",
        string_format("positive prompts file, one prompt per line (default: '%s')", params.cvector_positive_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.cvector_positive_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--negative-file"}, "FNAME",
        string_format("negative prompts file, one prompt per line (default: '%s')", params.cvector_negative_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.cvector_negative_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--pca-batch"}, "N",
        string_format("batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)", params.n_pca_batch),
        [](common_params & params, int value) {
            params.n_pca_batch = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--pca-iter"}, "N",
        string_format("number of iterations used for PCA (default: %d)", params.n_pca_iterations),
        [](common_params & params, int value) {
            params.n_pca_iterations = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--method"}, "{pca, mean}",
        "dimensionality reduction method to be used (default: pca)",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "pca") { params.cvector_dimre_method = DIMRE_METHOD_PCA; }
            else if (value == "mean") { params.cvector_dimre_method = DIMRE_METHOD_MEAN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--output-format"}, "{md,jsonl}",
        "output format for batched-bench results (default: md)",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "jsonl") { params.batched_bench_output_jsonl = true; }
            else if (value == "md") { params.batched_bench_output_jsonl = false; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));

    // llama-sweep-bench (examples/sweep-bench). Restores parsers the merge did not carry over into
    // arg.cpp -- -nrep, -wb, --output-format and --minilog from the legacy ik gpt_params_find_arg,
    // --sweep-stride and --sweep-memory from 1ce4bb973 (#2273) -- same fields, same accepted values.
    add_opt(common_arg(
        {"-nrep", "--n-repetitions"}, "N",
        string_format("number of repetitions for each context size (default: %d)", params.nrep),
        [](common_params & params, int value) {
            params.nrep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SWEEP_BENCH}));
    add_opt(common_arg(
        {"--sweep-stride"}, "N",
        string_format("measure every Nth sweep row (default: %d)", params.sweep_stride),
        [](common_params & params, int value) {
            params.sweep_stride = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SWEEP_BENCH}));
    add_opt(common_arg(
        {"--sweep-memory"},
        "report RSS high-water and sampled VRAM delta",
        [](common_params & params) {
            params.sweep_memory = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SWEEP_BENCH}));
    add_opt(common_arg(
        {"-wb", "--warmup-batch"},
        "run a warmup batch before measurement",
        [](common_params & params) {
            params.batch_warmup = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SWEEP_BENCH}));
    add_opt(common_arg(
        {"--output-format"}, "{md,jsonl}",
        "output format for sweep-bench results (default: md)",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "jsonl") { params.sweep_bench_output_jsonl = true; }
            else if (value == "md") { params.sweep_bench_output_jsonl = false; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_SWEEP_BENCH}));
    add_opt(common_arg(
        {"--minilog"},
        "print important information only",
        [](common_params & params) {
            params.minilog = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SWEEP_BENCH}));
    add_opt(common_arg(
        {"--log-disable"},
        "Log disable",
        [](common_params &) {
            common_log_pause(common_log_main());
        }
    ));
    add_opt(common_arg(
        {"--log-file"}, "FNAME",
        "Log to file",
        [](common_params &, const std::string & value) {
            common_log_set_file(common_log_main(), value.c_str());
        }
    ).set_env("LLAMA_ARG_LOG_FILE"));
    add_opt(common_arg(
        {"--log-jsonl"},
        {"--no-log-jsonl"},
        "Log as JSONL (one JSON object per line) to stdout, this also disables colored logging (default: disabled)",
        [](common_params &, bool value) {
            common_log_set_jsonl(value);
        }
    ).set_env("LLAMA_ARG_LOG_JSONL"));
    add_opt(common_arg(
        {"--log-prompts-dir"}, "PATH",
        "Log prompts to directory (auto-created if not present; only used for debugging, default: disabled)",
        [](common_params & params, const std::string & value) {
            params.path_prompts_log_dir = value;
            std::error_code ec;
            std::filesystem::create_directories(value, ec);
            if (ec) {
                fprintf(stderr, "warning: failed to create prompts-log-dir '%s': %s\n", value.c_str(), ec.message().c_str());
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--log-colors"}, "[on|off|auto]",
        "Set colored logging ('on', 'off', or 'auto', default: 'auto')\n"
        "'auto' enables colors when output is to a terminal",
        [](common_params &, const std::string & value) {
            if (is_truthy(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_ENABLED);
            } else if (is_falsey(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_DISABLED);
            } else if (is_autoy(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_AUTO);
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --log-colors: '%s'\n", value.c_str()));
            }
        }
    ).set_env("LLAMA_ARG_LOG_COLORS"));
    add_opt(common_arg(
        {"-v", "--verbose", "--log-verbose"},
        "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
        [](common_params & params) {
            params.verbosity = INT_MAX;
            common_log_set_verbosity_thold(INT_MAX);
        }
    ));
    add_opt(common_arg(
        {"--offline"},
        "Offline mode: forces use of cache, prevents network access",
        [](common_params & params) {
            params.offline = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_OFFLINE"));
    add_opt(common_arg(
        {"--download-connections"}, "N",
        "fetch large model files as byte ranges on N parallel connections; an interrupted download resumes at range granularity (default: 1 = single stream)",
        [](common_params & params, int value) {
            params.download_connections = std::max(1, std::min(value, 32));
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_DOWNLOAD_CONNECTIONS"));
    add_opt(common_arg(
        {"--download-chunk-mib"}, "N",
        "range size in MiB for --download-connections (default: 16; a file is never cut into more than 128 ranges)",
        [](common_params & params, int value) {
            params.download_chunk_mib = std::max(0, value);
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_DOWNLOAD, LLAMA_EXAMPLE_TOKENIZE}).set_env("LLAMA_ARG_DOWNLOAD_CHUNK_MIB"));
    add_opt(common_arg(
        {"-lv", "--verbosity", "--log-verbosity"}, "N",
        string_format("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\n"
            " - 0: generic output\n"
            " - 1: error\n"
            " - 2: warning\n"
            " - 3: info\n"
            " - 4: trace (more info)\n"
            " - 5: debug\n"
            "(default: %d)\n", params.verbosity),
        [](common_params & params, int value) {
            params.verbosity = value;
            common_log_set_verbosity_thold(value);
        }
    ).set_env("LLAMA_ARG_LOG_VERBOSITY"));
    add_opt(common_arg(
        {"--log-prefix"},
        {"--no-log-prefix"},
        "Enable prefix in log messages",
        [](common_params &, bool value) {
            common_log_set_prefix(common_log_main(), value);
        }
    ).set_env("LLAMA_ARG_LOG_PREFIX"));
    add_opt(common_arg(
        {"--log-timestamps"},
        {"--no-log-timestamps"},
        "Enable timestamps in log messages",
        [](common_params &, bool value) {
            common_log_set_timestamps(common_log_main(), value);
        }
    ).set_env("LLAMA_ARG_LOG_TIMESTAMPS"));
    add_opt(common_arg(
        // the fork's spelling (pre-merge server.cpp:62,470 @ 35f0cac75, field common_params.log_json
        // -- old server-common.cpp:26-65 server_log(), the fork's own MIT code) -- row S69-FORK-
        // SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, item 4. The fork's
        // server_log() was server.cpp-only; this codebase's logging is the one shared common_log
        // every binary uses, so -- like its --log-colors/--log-prefix/--log-timestamps siblings just
        // above -- this is registered generically rather than restricted to LLAMA_EXAMPLE_SERVER: any
        // binary can ask for JSON lines, and every binary defaults to today's plain-text output.
        {"--log-format"}, "{json,text}",
        "Set the log output format: 'json' emits one JSON object per line "
        "({\"timestamp\",\"level\",\"msg\"}), 'text' is the default prefixed format (default: 'text'). "
        "Takes effect immediately (like --log-colors/--log-prefix/--log-timestamps) and is also read "
        "from common_params.log_json by tools/server's main(), which re-applies it after its own log "
        "setup -- the fork's own server.cpp:470 read this same field the same way.",
        [](common_params & params, const std::string & value) {
            if (value == "json") {
                params.log_json = true;
            } else if (value == "text") {
                params.log_json = false;
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --log-format: '%s' (expected 'json' or 'text')\n", value.c_str()));
            }
            common_log_set_json_format(common_log_main(), params.log_json);
        }
    ).set_env("LLAMA_ARG_LOG_FORMAT"));

    //
    // speculative parameters
    //

    add_opt(common_arg(
        {"--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft"}, "<user>/<model>[:quant]",
        "Same as --hf-repo, but for the draft model (default: unused)",
        [](common_params & params, const std::string & value) {
            params.speculative.mparams_dft.hf_repo = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_HF_REPO"));
    add_opt(common_arg(
        {"--spec-draft-hf-revision", "--hf-revision-draft"}, "REV",
        "Same as --hf-revision, but for the draft model repo (default: unused)",
        [](common_params & params, const std::string & value) {
            params.speculative.mparams_dft.hf_revision = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_HF_REVISION"));
    add_opt(common_arg(
        {"--spec-draft-threads", "-td", "--threads-draft"}, "N",
        "number of threads to use during generation (default: same as --threads)",
        [](common_params & params, int value) {
            params.speculative.cpuparams.n_threads = value;
            if (params.speculative.cpuparams.n_threads <= 0) {
                params.speculative.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-threads-batch", "-tbd", "--threads-batch-draft"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads-draft)",
        [](common_params & params, int value) {
            params.speculative.cpuparams_batch.n_threads = value;
            if (params.speculative.cpuparams_batch.n_threads <= 0) {
                params.speculative.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-mask", "-Cd", "--cpu-mask-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.speculative.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.speculative.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-range", "-Crd", "--cpu-range-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft",
        [](common_params & params, const std::string & range) {
            params.speculative.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.speculative.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-placement", "--cpu-placement-draft"}, "{none,physical}",
        "Draft model CPU worker placement when no -Cd/-Crd mask is given (default: same as --cpu-placement)",
        [](common_params & params, const std::string & value) {
            bool physical;
            if (!common_cpu_placement_parse(value, physical)) {
                throw std::invalid_argument("invalid --cpu-placement-draft value '" + value + "' (expected none or physical)");
            }
            params.speculative.cpuparams.place_physical       = physical;
            params.speculative.cpuparams_batch.place_physical = physical;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-strict", "--cpu-strict-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: same as --cpu-strict)",
        [](common_params & params, int value) {
            params.speculative.cpuparams.strict_cpu = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-prio", "--prio-draft"}, "N",
        string_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.speculative.cpuparams.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-poll", "--poll-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: same as --poll)",
        [](common_params & params, int value) {
            params.speculative.cpuparams.poll = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-mask-batch", "-Cbd", "--cpu-mask-batch-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.speculative.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.speculative.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-range-batch", "-Crbd", "--cpu-range-batch-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft-batch)",
        [](common_params & params, const std::string & range) {
            params.speculative.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.speculative.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--spec-draft-cpu-strict-batch", "--cpu-strict-batch-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: --cpu-strict-draft)",
        [](common_params & params, int value) {
            params.speculative.cpuparams_batch.strict_cpu = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-prio-batch", "--prio-batch-draft"}, "N",
        string_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.speculative.cpuparams_batch.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-poll-batch", "--poll-batch-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: --poll-draft)",
        [](common_params & params, int value) {
            params.speculative.cpuparams_batch.poll = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-type-k", "-ctkd", "--cache-type-k-draft"}, "TYPE",
        string_format(
            "KV cache data type for K for the draft model\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            // ADAPTED (S69-ARGCPP-SPEC-SHAPE-MISMATCH-2026-09-08-01): header's
            // common_params_speculative::cache_type_k is std::string (raw name), not
            // ggml_type -- conversion happens at point of use (common_model_params_to_llama/
            // common_context_params_to_llama), same pattern as top-level params.cache_type_k.
            params.speculative.cache_type_k.empty() ? "f16" : params.speculative.cache_type_k.c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.speculative.cache_type_k = value;
        }
    ).set_env("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K"));
    add_opt(common_arg(
        {"--spec-draft-type-v", "-ctvd", "--cache-type-v-draft"}, "TYPE",
        string_format(
            "KV cache data type for V for the draft model\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            params.speculative.cache_type_v.empty() ? "f16" : params.speculative.cache_type_v.c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.speculative.cache_type_v = value;
        }
    ).set_env("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V"));
    add_opt(common_arg(
        {"--spec-draft-override-tensor", "-otd", "--override-tensor-draft"}, "<tensor name pattern>=<buffer type>,...",
        "override tensor buffer type for draft model", [](common_params & params, const std::string & value) {
            parse_tensor_buffer_overrides(value, params.speculative.tensor_buft_overrides);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-moe", "-cmoed", "--cpu-moe-draft"},
        "keep all Mixture of Experts (MoE) weights in the CPU for the draft model",
        [](common_params & params) {
            params.speculative.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_CPU_MOE"));
    add_opt(common_arg(
        {"--spec-draft-n-cpu-moe", "--spec-draft-ncmoe", "-ncmoed", "--n-cpu-moe-draft"}, "N",
        "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU for the draft model",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            for (int i = 0; i < value; ++i) {
                static std::list<std::string> buft_overrides_draft;
                buft_overrides_draft.push_back(llm_ffn_exps_block_regex(i));
                params.speculative.tensor_buft_overrides.push_back({buft_overrides_draft.back().c_str(), ggml_backend_cpu_buffer_type()});
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE"));

    add_opt(common_arg(
        {"--spec-draft-n-max"}, "N",
        string_format("number of tokens to draft for speculative decoding (default: %d)", params.speculative.n_max),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.n_max = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_N_MAX"));
    add_opt(common_arg(
        {"--spec-draft-n-min"}, "N",
        string_format("minimum number of draft tokens to use for speculative decoding (default: %d)", params.speculative.n_min),
        [](common_params & params, int value) {
            params.speculative.n_min = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_N_MIN"));
    add_opt(common_arg(
        {"--spec-draft-n-ubatch"}, "N",
        string_format("physical maximum batch size of the draft context, use a smaller value to spare VRAM (default: %d - same as --n-ubatch)", params.speculative.n_ubatch),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.n_ubatch = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_N_UBATCH"));
    add_opt(common_arg(
        {"--spec-synth-len"}, "L",
        "target mean synthetic acceptance length, including the target token (benchmarking only)",
        [](common_params & params, const std::string & value) {
            const std::string text = string_strip(value);
            size_t pos = 0;
            const double length = std::stod(text, &pos);
            if (pos != text.size() || length == -1.0) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.synth_len = length;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_SYNTH_LEN"));
    add_opt(common_arg(
        {"--spec-synth-rates"}, "P0,P1,...",
        "comma-separated unconditional per-position synthetic acceptance probabilities (benchmarking only)",
        [](common_params & params, const std::string & value) {
            const auto values = string_split<std::string>(value, ',');
            std::vector<double> rates;
            rates.reserve(values.size());
            for (const auto & raw : values) {
                const std::string text = string_strip(raw);
                size_t pos = 0;
                const double rate = std::stod(text, &pos);
                if (pos != text.size()) {
                    throw std::invalid_argument("invalid value");
                }
                rates.push_back(rate);
            }
            params.speculative.synth_rates = std::move(rates);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_SYNTH_RATES"));

    add_opt(common_arg(
        {"--spec-draft-p-split", "--draft-p-split"}, "P",
        string_format("speculative decoding split probability (default: %.2f)", (double)params.speculative.p_split),
        [](common_params & params, const std::string & value) {
            params.speculative.p_split = std::stof(value);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_P_SPLIT"));
    add_opt(common_arg(
        {"--spec-draft-p-min", "--draft-p-min"}, "P",
        string_format("minimum speculative decoding probability (greedy): a draft stops at the first token whose "
                      "normalised probability is below P (default: %.2f for MTP; draft-model speculation applies it only "
                      "when given, otherwise 0 = no early stop, as upstream)", (double)params.speculative.p_min),
        [](common_params & params, const std::string & value) {
            params.speculative.p_min = std::stof(value);
            params.speculative.p_min_explicit = true;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_P_MIN"));
    // DISABLE-REVERSIBLE: new opt-in switch, off by default until the acceptance-rate measurement on the row
    // S69-SPEC-DECODE-REJECTION-SAMPLING-ACCEPTANCE-PICK-2026-09-28-01 says which default wins; nothing existing is turned off.
    // RESEARCH-DONE: Leviathan et al. 2023 (arXiv:2211.17192) Thm 3.5; Chen et al. 2023 (arXiv:2302.01318).
    add_opt(common_arg(
        {"--spec-draft-sample"},
        {"--no-spec-draft-sample"},
        string_format("draft model samples from its distribution q and the server verifies by rejection sampling "
                      "(accept min(1,p/q), resample the residual) instead of drafting the argmax (default: %s)",
                      params.speculative.draft_sample ? "on" : "off"),
        [](common_params & params, bool value) {
            params.speculative.draft_sample = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SPEC_DRAFT_SAMPLE"));
    // DISABLE-REVERSIBLE: new opt-in switch; nothing existing is turned off. It costs a full-vocab grammar pass per
    // drafted token on the draft model, so it stays opt-in until a model-level measurement says the acceptance gain pays
    // for it (row S69-LMDEPLOY-PICK-GRAMMAR-MASKED-SPEC-DRAFT-2026-10-18-01).
    // RESEARCH-DONE: LMDeploy v0.17.0 (Apache-2.0) lmdeploy/pytorch/spec_decode/guided_spec_helper.py:69-98, :124-167,
    // :221-231 masks each draft step with a forked grammar matcher; read, rewritten as a grammar copy, no code copied.
    add_opt(common_arg(
        {"--spec-draft-grammar"},
        {"--no-spec-draft-grammar"},
        string_format("draft model drafts under a copy of the request's grammar, so drafts the grammar forbids are not "
                      "proposed (draft-model speculation; with --spec-draft-sample the draft is drawn from the grammar-masked, "
                      "renormalised distribution; default: %s)",
                      params.speculative.draft_grammar ? "on" : "off"),
        [](common_params & params, bool value) {
            params.speculative.draft_grammar = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SPEC_DRAFT_GRAMMAR"));
    // DISABLE-REVERSIBLE: default ON because it is lossless (a token the grammar forbids cannot be what the constrained target
    // samples) and only shortens the verify batch; --no-spec-draft-validate restores the untrimmed drafts. No effect without a
    // grammar / json_schema / choice constraint on the request. Row S69-VLLM-W2-GRAMMAR-VALIDATED-DRAFTS-2026-10-27-01.
    // RESEARCH-DONE: vLLM 924707f1bf94 vllm/v1/structured_output/__init__.py:294-312 validate_tokens, scheduler.py:2487,2515,
    // backend_guidance.py:185-200 (Apache-2.0); read, rewritten as a walk over a copy of the slot's grammar, no code copied.
    add_opt(common_arg(
        {"--spec-draft-validate"},
        {"--no-spec-draft-validate"},
        string_format("cut every speculative draft to its grammar-valid prefix before verification when the request has a grammar "
                      "(any proposer; lossless; default: %s)",
                      params.speculative.draft_validate ? "on" : "off"),
        [](common_params & params, bool value) {
            params.speculative.draft_validate = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SPEC_DRAFT_VALIDATE"));
    // DISABLE-REVERSIBLE: default OFF (opt-in); --no-spec-async or omitting the flag is the serial path, byte for byte.
    // RESEARCH-DONE: PEARL arXiv 2408.11850 (pre/post-verify), AMUSD arXiv 2410.17375 (continuous loops + rollback), read in
    // S69_ASYM_SPEC_PIPELINES_AND_KV_COMPRESSION_2026-11-06.md; no llama.cpp/ggml implementation exists (web search 2026-11-06); own design.
    add_opt(common_arg(
        {"--spec-async"},
        {"--no-spec-async"},
        string_format("overlap the draft-model drafter with the target verify on a worker thread (draft-model speculation, greedy "
                      "draft; needs a draft context that can roll back; lossless; default: %s)",
                      params.speculative.async ? "on" : "off"),
        [](common_params & params, bool value) {
            params.speculative.async = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SPEC_ASYNC"));
    add_opt(common_arg(
        {"--spec-async-wait-ms"}, "N",
        string_format("--spec-async: milliseconds the lookahead may keep running after the target verify finished (default: %d = stop at "
                      "the next token boundary)", params.speculative.async_wait_ms),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.async_wait_ms = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SPEC_ASYNC_WAIT_MS"));
    // DISABLE-REVERSIBLE: default OFF; omitting the flag keeps the wait_ms join byte for byte.
    // RESEARCH-DONE: PEARL arXiv 2408.11850 sec. 3.4 (post-verify: the draft model is not interrupted while the verify decides), own design;
    // S69_SPEC_ASYNC_LOOKAHEAD_TRUNCATION_2026-11-06.md sec. 5 option O1.
    add_opt(common_arg(
        {"--spec-async-informed-join"},
        {"--no-spec-async-informed-join"},
        string_format("--spec-async: decide the join from the verify result for greedy requests: keep the lookahead running when the whole "
                      "draft was accepted and the target's token equals its first token, stop it at once otherwise; "
                      "--spec-async-wait-ms (when > 0) stays an upper cap (default: %s)",
                      params.speculative.async_informed ? "on" : "off"),
        [](common_params & params, bool value) {
            params.speculative.async_informed = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SPEC_ASYNC_INFORMED_JOIN"));
    add_opt(common_arg(
        {"--spec-draft-backend-sampling"},
        {"--no-spec-draft-backend-sampling"},
        string_format("offload draft sampling to the backend (default: %s)",
                      params.speculative.backend_sampling ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.speculative.backend_sampling = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING"));
    add_opt(common_arg(
        {"--spec-draft-sampling"}, "{greedy,probabilistic}",
        string_format("how the draft is sampled: greedy takes its argmax, probabilistic samples it and has "
                      "the target verify by rejection sampling (default: %s)",
                      params.speculative.draft_sample ? "probabilistic" : "greedy"),
        [](common_params & params, const std::string & value) {
            if (value == "greedy") {
                params.speculative.draft_sample = false;
            } else if (value == "probabilistic") {
                params.speculative.draft_sample = true;
            } else {
                throw std::invalid_argument("invalid value, must be one of: greedy, probabilistic");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_SAMPLING"));
    add_opt(common_arg(
        {"--spec-draft-device", "-devd", "--device-draft"}, "<dev1,dev2,..>",
        "comma-separated list of devices to use for offloading the draft model (none = don't offload, default: follows --device)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            // ADAPTED (S69-ARGCPP-SPEC-SHAPE-MISMATCH-2026-09-08-01): header's
            // common_params_speculative::devices is a flat std::string (raw CSV), not a
            // std::vector<ggml_backend_dev_t> -- store the raw value, parse_device_list()
            // is for callers that need the resolved device-pointer list at use time.
            params.speculative.devices = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    GGML_ASSERT(params.speculative.n_gpu_layers < 0); // string_format would need to be extended for a default >= 0
    add_opt(common_arg(
        {"--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}, "N",
        string_format("max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)",
            params.speculative.n_gpu_layers == -1 ? "auto" : "all"),
        [](common_params & params, const std::string & value) {
            if (value == "auto") {
                params.speculative.n_gpu_layers = -1;
            } else if (value == "all") {
                params.speculative.n_gpu_layers = -2;
            } else {
                params.speculative.n_gpu_layers = std::stoi(value);
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: no usable GPU found, --gpu-layers-draft option will be ignored\n");
                fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\n");
                fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_N_GPU_LAYERS_DRAFT"));
    add_opt(common_arg(
        {"--spec-draft-model", "-md", "--model-draft"}, "FNAME",
        "draft model for speculative decoding (default: unused)",
        [](common_params & params, const std::string & value) {
            params.speculative.mparams_dft.path = value;
            params.speculative.mparams_dft.hf_file = value; // will be used if --spec-draft-hf is set
        }
    // LLAMA_EXAMPLE_IMATRIX ADDED (row S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-
    // BUILT-TOOLS-2026-09-27-01): the paired draft-model imatrix path needs -md reachable here
    // too, to name the Gemma4-MTP-assistant draft model paired against the target. Does not
    // touch the flag literal itself (already REGISTERED in .fork-cli-flags-baseline.txt), only
    // widens which examples parse it.
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_IMATRIX}).set_env("LLAMA_ARG_SPEC_DRAFT_MODEL"));
    // WIRING A FEATURE THAT WAS ALREADY BUILT HERE AND COULD NOT BE REACHED.
    // (row S69-SPEC-REPLACE-BUILT-BUT-UNREACHABLE-2026-09-14-01)
    //
    // The whole translation path already existed in this tree and was complete:
    //   common/common.h:401                 std::vector<std::pair<std::string,std::string>> replacements
    //   common/speculative.cpp:2306         config.params.replacements passed into the state ctor
    //   common/speculative.cpp:430-460      ctor builds vocab_map from those pairs when the vocabs
    //                                       are incompatible
    //   common/speculative.cpp:494-518      detokenize(target) -> replace_to_dft -> tokenize(draft),
    //                                       applied to the prompt AND to id_last
    //   common/speculative.cpp:650-657      the reverse on the way back out
    // Every piece was present. NOTHING EVER WROTE TO `replacements`, because no command-line option
    // existed to populate it, so vocab_map was permanently empty and replace_to_dft/replace_to_tgt
    // were identity functions on every run. The feature was dead on arrival for anyone not calling
    // the C++ API directly and setting the field by hand.
    //
    // Found 2026-09-14 by diffing our llama-cli --help against upstream's: --spec-replace was one
    // of only 8 flags upstream had that we did not, and the only one that turned out to be a real
    // gap rather than a rename or a build-flag difference. The gap was three lines of registration,
    // not a port -- PRESENT IS NOT WORKING, measured again.
    //
    // Registration adapted from upstream llama.cpp (MIT), common/arg.cpp, which spells it the same
    // way; the .set_spec() call and the option's placement follow THIS tree's convention for the
    // --spec-* family rather than upstream's layout.
    add_opt(common_arg(
        {"--spec-replace"}, "TARGET", "DRAFT",
        "translate the string in TARGET into DRAFT if the draft model and main model are not compatible",
        [](common_params & params, const std::string & tgt, const std::string & dft) {
            params.speculative.replacements.push_back({ tgt, dft });
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-type"}, common_speculative_all_types_str(),
        // ADAPTED (S69-ARGCPP-SPEC-SHAPE-MISMATCH-2026-09-08-01): theirs printed
        // params.speculative.types (missing vector) directly; built from the resolved stage
        // chain instead -- same set of types get_resolved_stages() would report.
        string_format("comma-separated list of types of speculative decoding to use (default: %s)\n",
            common_speculative_type_name_str([&params]{
                std::vector<common_speculative_type> t;
                for (const auto & st : params.speculative.get_resolved_stages()) t.push_back(st.type);
                return t;
            }()).c_str()),
        [](common_params & params, const std::string & value) {
            spec_type_arg_apply(params, value);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_TYPE"));
    // ADAPTED, all 4 ngram flag groups below (mod/simple/map-k/map-k4v)
    // (S69-ARGCPP-SPEC-SHAPE-MISMATCH-2026-09-08-01): theirs read/wrote 4 distinct nested
    // sub-structs (.ngram_mod{n_min,n_max,n_match}, .ngram_simple/.ngram_map_k/.ngram_map_k4v
    // each {size_n,size_m,min_hits}), none of which exist on the staged header. The header
    // carries ONE flat triple instead (ngram_size_n/ngram_size_m/ngram_min_hits on
    // common_params_speculative) that with_stage_overrides() (common.cpp) applies to whichever
    // stage is active -- by design, only one ngram-family stage config is "current" at a time.
    // All 4 groups below are mapped onto that same flat triple (n_min/size_n->ngram_size_n,
    // n_max/size_m->ngram_size_m, n_match/min_hits->ngram_min_hits); the currently-selected
    // --spec-type stage decides which algorithm actually consumes them.
    add_opt(common_arg(
        {"--spec-ngram-mod-n-min"}, "N",
        string_format("minimum number of ngram tokens to use for ngram-based speculative decoding (default: %d)", params.speculative.ngram_size_n),
        [](common_params & params, int value) {
            if (value < 0 || value > 1024) {
                throw std::invalid_argument("ngram n-min must be between 0 and 1024 inclusive");
            }
            params.speculative.ngram_size_n = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-mod-n-max"}, "N",
        string_format("maximum number of ngram tokens to use for ngram-based speculative decoding (default: %d)", params.speculative.ngram_size_m),
        [](common_params & params, int value) {
            if (value < 0 || value > 1024) {
                throw std::invalid_argument("ngram n-max must be between 0 and 1024 inclusive");
            }
            params.speculative.ngram_size_m = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-mod-n-match"}, "N",
        string_format("ngram-mod lookup length (default: %d)", params.speculative.ngram_min_hits),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_min_hits = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-ngram-simple-size-n"}, "N",
        string_format("ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram_size_n),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_size_n = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-simple-size-m"}, "N",
        string_format("ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram_size_m),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_size_m = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-simple-min-hits"}, "N",
        string_format("minimum hits for ngram-simple speculative decoding (default: %d)", params.speculative.ngram_min_hits),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("ngram min hits must be at least 1");
            }
            params.speculative.ngram_min_hits = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-ngram-map-k-size-n"}, "N",
        string_format("ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram_size_n),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_size_n = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k-size-m"}, "N",
        string_format("ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram_size_m),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_size_m = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k-min-hits"}, "N",
        string_format("minimum hits for ngram-map-k speculative decoding (default: %d)", params.speculative.ngram_min_hits),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("ngram min hits must be at least 1");
            }
            params.speculative.ngram_min_hits = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-ngram-map-k4v-size-n"}, "N",
        string_format("ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram_size_n),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_size_n = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k4v-size-m"}, "N",
        string_format("ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram_size_m),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_size_m = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k4v-min-hits"}, "N",
        string_format("minimum hits for ngram-map-k4v speculative decoding (default: %d)", params.speculative.ngram_min_hits),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("ngram min hits must be at least 1");
            }
            params.speculative.ngram_min_hits = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    //
    // removed params
    //

    add_opt(common_arg(
        {"--draft", "--draft-n", "--draft-max"}, "N",
        "the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use --spec-draft-n-max or --spec-ngram-mod-n-max");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_DRAFT_MAX"));
    add_opt(common_arg(
        {"--draft-min", "--draft-n-min"}, "N",
        "the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use --spec-draft-n-min or --spec-ngram-mod-n-min");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_DRAFT_MIN"));
    add_opt(common_arg(
        {"--spec-ngram-size-n"}, "N",
        "the argument has been removed. use the respective --spec-ngram-*-size-n or --spec-ngram-mod-n-match",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use the respective --spec-ngram-*-size-n");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--spec-ngram-size-m"}, "N",
        "the argument has been removed. use the respective --spec-ngram-*-size-m",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use the respective --spec-ngram-*-size-m");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--spec-ngram-min-hits"}, "N",
        "the argument has been removed. use the respective --spec-ngram-*-min-hits",
        [](common_params & /*params*/, int /*value*/) {
            arg_removed("use the respective --spec-ngram-*-min-hits");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}));

    //
    // TTS params
    //

    add_opt(common_arg(
        {"--tts-lang"}, "FNAME",
        "language (ISO 639-1) for audio generation\n"
        "see tts/README.md for per-model usage notes",
        [](common_params & params, const std::string & value) {
            params.tts_lang = value;
        }
    ).set_examples({LLAMA_EXAMPLE_TTS}));
    add_opt(common_arg(
        {"--tts-speaker-file"}, "FNAME",
        "speaker file path for audio generation",
        [](common_params & params, const std::string & value) {
            params.tts_speaker_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_TTS}));

    //
    // diffusion params
    //

    add_opt(common_arg(
        {"--diffusion-steps"}, "N",
        string_format("number of diffusion steps (default: %d)", params.diffusion.steps),
        [](common_params & params, int value) { params.diffusion.steps = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-blocks"}, "N",
        string_format("max block-autoregressive blocks for block-diffusion models (default: %d)", params.diffusion.blocks),
        [](common_params & params, int value) { params.diffusion.blocks = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-visual"},
        string_format("enable visual diffusion mode (show progressive generation) (default: %s)", params.diffusion.visual_mode ? "true" : "false"),
        [](common_params & params) { params.diffusion.visual_mode = true; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-visual-progress"},
        string_format("show the step progress bar in visual mode (default: %s)", params.diffusion.visual_progress ? "true" : "false"),
        [](common_params & params) { params.diffusion.visual_progress = true; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-visual-interval"}, "N",
        string_format("redraw the visual canvas every Nth step; all steps are still computed (default: %d)", params.diffusion.visual_interval),
        [](common_params & params, int value) { params.diffusion.visual_interval = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eps"}, "F",
        string_format("epsilon for timesteps (default: %.6f)", (double) params.diffusion.eps),
        [](common_params & params, const std::string & value) { params.diffusion.eps = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-algorithm"}, "N",
        string_format(
            "diffusion algorithm: 0=DIFFUSION_ALGORITHM_ORIGIN, 1=DIFFUSION_ALGORITHM_ENTROPY_BASED, "
            "2=DIFFUSION_ALGORITHM_MARGIN_BASED, 3=DIFFUSION_ALGORITHM_RANDOM, "
            "4=DIFFUSION_ALGORITHM_CONFIDENCE_BASED (default: %d)", params.diffusion.algorithm),
        [](common_params & params, int value) { params.diffusion.algorithm = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-alg-temp"}, "F",
        string_format("dream algorithm temperature (default: %.3f)", (double) params.diffusion.alg_temp),
        [](common_params & params, const std::string & value) { params.diffusion.alg_temp = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-block-length"}, "N",
        string_format("llada block length for generation (default: %d)", params.diffusion.block_length),
        [](common_params & params, int value) { params.diffusion.block_length = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-cfg-scale"}, "F",
        string_format("llada classifier-free guidance scale (default: %.3f)", (double) params.diffusion.cfg_scale),
        [](common_params & params, const std::string & value) { params.diffusion.cfg_scale = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-add-gumbel-noise"}, "F",
        string_format("add gumbel noise to the logits if temp > 0.0 (default: %s)", params.diffusion.add_gumbel_noise ? "true" : "false"),
        [](common_params & params, const std::string & value) { params.diffusion.add_gumbel_noise = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eb"}, "MODE",
        "entropy-bound decoder for canvas/block-diffusion models (DiffusionGemma): auto|on|off (default: auto)",
        [](common_params & params, const std::string & value) {
            if      (value == "off")  { params.diffusion.eb_mode = 2; }
            else if (value == "on")   { params.diffusion.eb_mode = 1; }
            else if (value == "auto") { params.diffusion.eb_mode = 0; }
            else { throw std::invalid_argument("--diffusion-eb must be auto|on|off"); }
        }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eb-t-min"}, "F",
        "entropy-bound: temperature at the last step (default: from model metadata, else 0.4)",
        [](common_params & params, const std::string & value) { params.diffusion.eb_t_min = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eb-t-max"}, "F",
        "entropy-bound: temperature at the first step (default: from model metadata, else 0.8)",
        [](common_params & params, const std::string & value) { params.diffusion.eb_t_max = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eb-entropy-bound"}, "F",
        "entropy-bound: accept lowest-entropy tokens within this MI bound (default: from model metadata, else 0.1)",
        [](common_params & params, const std::string & value) { params.diffusion.eb_entropy_bound = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eb-stability"}, "N",
        "entropy-bound: steps the argmax canvas must hold to stop (default: from model metadata, else 1)",
        [](common_params & params, int value) { params.diffusion.eb_stability = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eb-confidence"}, "F",
        "entropy-bound: stop once mean canvas entropy drops below this (default: from model metadata, else 0.005)",
        [](common_params & params, const std::string & value) { params.diffusion.eb_confidence = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eb-max-steps"}, "N",
        "entropy-bound: max denoising steps (default: from model metadata, else 48)",
        [](common_params & params, int value) { params.diffusion.eb_max_steps = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-kv-cache"}, "MODE",
        "entropy-bound: prefix KV cache (PREFILL prompt once, decode canvas-only per step): auto|on|off "
        "(default: auto = on for single-GPU canvas models)",
        [](common_params & params, const std::string & value) {
            if      (value == "off")  { params.diffusion.eb_kv_cache = 2; }
            else if (value == "on")   { params.diffusion.eb_kv_cache = 1; }
            else if (value == "auto") { params.diffusion.eb_kv_cache = 0; }
            else { throw std::invalid_argument("--diffusion-kv-cache must be auto|on|off"); }
        }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-gpu-sampling"}, "MODE",
        "entropy-bound: device-resident self-conditioning (keep prev-step canvas logits on-device, no "
        "per-step host upload): auto|on|off (default: auto = on for single-GPU canvas models)",
        [](common_params & params, const std::string & value) {
            if      (value == "off")  { params.diffusion.eb_gpu_sampling = 2; }
            else if (value == "on")   { params.diffusion.eb_gpu_sampling = 1; }
            else if (value == "auto") { params.diffusion.eb_gpu_sampling = 0; }
            else { throw std::invalid_argument("--diffusion-gpu-sampling must be auto|on|off"); }
        }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-gpu-sample-reduce"}, "MODE",
        "entropy-bound: Stage-1 device-side argmax/entropy/sampling reduction read straight from sc_dev "
        "(needs --diffusion-gpu-sampling on): auto|on|off (default: auto = on for single-GPU canvas)",
        [](common_params & params, const std::string & value) {
            if      (value == "off")  { params.diffusion.eb_gpu_sample_reduce = 2; }
            else if (value == "on")   { params.diffusion.eb_gpu_sample_reduce = 1; }
            else if (value == "auto") { params.diffusion.eb_gpu_sample_reduce = 0; }
            else { throw std::invalid_argument("--diffusion-gpu-sample-reduce must be auto|on|off"); }
        }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        { "-lr", "--learning-rate" }, "ALPHA",
        string_format("adamw or sgd optimizer alpha (default: %.2g); note: sgd alpha recommended ~10x (no momentum)", (double) params.lr.lr0),
        [](common_params & params, const std::string & value) { params.lr.lr0 = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg({ "-lr-min", "--learning-rate-min" }, "ALPHA",
        string_format("(if >0) final learning rate after decay (if -decay-epochs is set, default=%.2g)",
            (double) params.lr.lr_min),
        [](common_params & params, const std::string & value) { params.lr.lr_min = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-decay-epochs", "--learning-rate-decay-epochs"}, "ALPHA",
        string_format("(if >0) decay learning rate to -lr-min after this many epochs (exponential decay, default=%.2g)", (double) params.lr.decay_epochs),
        [](common_params & params, const std::string & value) { params.lr.decay_epochs = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-wd", "--weight-decay"}, "WD",
        string_format("adamw or sgd optimizer weight decay (0 is off; recommend very small e.g. 1e-9) (default: %.2g).", (double) params.lr.wd),
        [](common_params & params, const std::string & value) { params.lr.wd = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-val-split", "--val-split"}, "FRACTION",
        string_format("fraction of data to use as validation set for training (default: %.2g).", (double) params.val_split),
        [](common_params & params, const std::string & value) { params.val_split = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-epochs", "--epochs"}, "N",
        string_format("optimizer max # of epochs (default: %d)", params.lr.epochs),
        [](common_params & params, int epochs) { params.lr.epochs = epochs; }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-opt", "--optimizer"}, "sgd|adamw", "adamw or sgd",
        [](common_params & params, const std::string & name) {
            params.optimizer = common_opt_get_optimizer(name.c_str());
            if (params.optimizer == GGML_OPT_OPTIMIZER_TYPE_COUNT) {
                throw std::invalid_argument("invalid --optimizer, valid options: adamw, sgd");
            }
        }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"--check"},
        string_format("check rather than generate results (default: %s)", params.check ? "true" : "false"),
        [](common_params & params) {
            params.check = true;
        }
    ).set_examples({LLAMA_EXAMPLE_RESULTS}));
    add_opt(common_arg(
        {"--save-logits"},
        string_format("save final logits to files for verification (default: %s)", params.save_logits ? "true" : "false"),
        [](common_params & params) {
            params.save_logits = true;
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--logits-output-dir"}, "PATH",
        string_format("directory for saving logits output files (default: %s)", params.logits_output_dir.c_str()),
        [](common_params & params, const std::string & value) {
            params.logits_output_dir = value;
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--tensor-filter"}, "REGEX",
        "filter tensor names for debug output (regex pattern, can be specified multiple times)",
        [](common_params & params, const std::string & value) {
            params.tensor_filter.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));

    // presets

    add_opt(common_arg(
        {"--embd-gemma-default"},
        string_format("use default EmbeddingGemma model (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF";
            params.model.hf_file = "embeddinggemma-300M-qat-Q4_0.gguf";
            params.port = 8011;
            params.n_ubatch = 2048;
            params.n_batch = 2048;
            params.n_parallel = 32;
            params.n_ctx = 2048*params.n_parallel;
            params.verbose_prompt = true;
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-1.5b-default"},
        string_format("use default Qwen 2.5 Coder 1.5B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-1.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-3b-default"},
        string_format("use default Qwen 2.5 Coder 3B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-3b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-7b-default"},
        string_format("use default Qwen 2.5 Coder 7B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-7b-spec"},
        string_format("use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.speculative.mparams_dft.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.mparams_dft.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-14b-spec"},
        string_format("use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-14B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-14b-q8_0.gguf";
            params.speculative.mparams_dft.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.mparams_dft.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-30b-default"},
        string_format("use default Qwen 3 Coder 30B A3B Instruct (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8_0-GGUF";
            params.model.hf_file = "qwen3-coder-30b-a3b-instruct-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--gpt-oss-20b-default"},
        string_format("use gpt-oss-20b (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gpt-oss-20b-GGUF";
            params.model.hf_file = "gpt-oss-20b-mxfp4.gguf";
            params.port = 8013;
            params.n_ubatch = 2048;
            params.n_batch = 32768;
            params.n_parallel = 2;
            params.n_ctx = 131072*params.n_parallel;
            params.sampling.temp = 1.0f;
            params.sampling.top_p = 1.0f;
            params.sampling.top_k = 0;
            params.sampling.min_p = 0.01f;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--gpt-oss-120b-default"},
        string_format("use gpt-oss-120b (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gpt-oss-120b-GGUF";
            params.port = 8013;
            params.n_ubatch = 2048;
            params.n_batch = 32768;
            params.n_parallel = 2;
            params.n_ctx = 131072*params.n_parallel;
            params.sampling.temp = 1.0f;
            params.sampling.top_p = 1.0f;
            params.sampling.top_k = 0;
            params.sampling.min_p = 0.01f;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--vision-gemma-4b-default"},
        string_format("use Gemma 3 4B QAT (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gemma-3-4b-it-qat-GGUF";
            params.port = 8014;
            params.n_ctx = 0;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--vision-gemma-12b-default"},
        string_format("use Gemma 3 12B QAT (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gemma-3-12b-it-qat-GGUF";
            params.port = 8014;
            params.n_ctx = 0;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-default"},
        string_format("enable default speculative decoding config"),
        // HOOK-OVERRIDE: sst -- the "not sure if this is a good config" line below is upstream's
        // own pre-existing inline note (unchanged wording, carried through this shape-mismatch
        // adaptation), not a new task/plan marker introduced here.
        [](common_params & params) {
            // ADAPTED (S69-ARGCPP-SPEC-SHAPE-MISMATCH-2026-09-08-01): theirs' nested
            // params.speculative.ngram_mod.{n_match,n_min,n_max} sub-struct does not exist on
            // the staged header (that name is taken by a live std::shared_ptr<common_ngram_mod>
            // used by the real NGRAM_MOD engine in speculative.cpp -- see common.h). Mapped onto
            // the header's flat ngram_size_n/ngram_size_m/ngram_min_hits, the fields the
            // resolved-stage chain (with_stage_overrides, common.cpp) and the real engine
            // (common_speculative_init's ngram_size_n use) actually read. n_min->ngram_size_n
            // (lookup length, the one field the live engine's common_ngram_mod ctor consumes
            // today) is the load-bearing translation; n_max->ngram_size_m and n_match->
            // ngram_min_hits are stored for CLI completeness but the current common_ngram_mod
            // ctor (common/ngram-mod.h) does not yet read them -- BENCH-PENDING, not silently
            // claimed functional.
            spec_stages_push(params, COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
            params.speculative.ngram_min_hits = 24;
            params.speculative.ngram_size_n   = 48;
            params.speculative.ngram_size_m   = 64;

            // upstream note, unchanged: not sure if this is a good config - explore more
            // settings and potentially enable it
            //spec_stages_push(params, COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
            //params.speculative.ngram_size_n   = 8;
            //params.speculative.ngram_size_m   = 24;
            //params.speculative.ngram_min_hits = 2;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    // CATS activation sparsity (S69-CATS-STAGE1-2026-09-28). Training-free magnitude threshold
    // on the SwiGLU gate activation, arXiv:2404.08763 Sec 3.1. Stage 1: CPU-only, dense-mask
    // quality path (no sparse matvec kernel yet -- that is a later GPU stage per the task row).
    add_opt(common_arg(
        {"--cats-sparsity"}, "F",
        string_format("CATS activation sparsity target in [0,1); 0 = off (default: %.3f). Requires --cats-thresholds.",
                       params.cats_sparsity),
        [](common_params & params, const std::string & value) {
        float f = std::stof(value);
        if (f < 0.0f || f >= 1.0f) {
            throw std::invalid_argument("--cats-sparsity must be in [0, 1)");
        }
        params.cats_sparsity = f;
        return;}
    ));

    add_opt(common_arg(
        {"--cats-thresholds"}, "FILE",
        "path to the per-layer threshold JSON produced by tools/cats-calibrate (required when --cats-sparsity > 0)",
        [](common_params & params, const std::string & value) {
        params.cats_thresholds_file = value;
        return;}
    ));

    // TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28). Training-free INPUT magnitude
    // threshold applied before Wq/Wk/Wv/Wgate/Wup/Wdown, arXiv:2408.14690 Sec 4.2 Definition 1.
    // Stage 1: CPU-only, dense-mask quality path; Wo not wired this session (see llama.h).
    add_opt(common_arg(
        {"--teal-sparsity"}, "F",
        string_format("TEAL activation sparsity target in [0,1); 0 = off (default: %.3f). Requires --teal-thresholds.",
                       params.teal_sparsity),
        [](common_params & params, const std::string & value) {
        float f = std::stof(value);
        if (f < 0.0f || f >= 1.0f) {
            throw std::invalid_argument("--teal-sparsity must be in [0, 1)");
        }
        params.teal_sparsity = f;
        return;}
    ));

    add_opt(common_arg(
        {"--teal-thresholds"}, "FILE",
        "path to the per-layer 3-family threshold JSON produced by tools/cats-calibrate --teal (required when --teal-sparsity > 0)",
        [](common_params & params, const std::string & value) {
        params.teal_thresholds_file = value;
        return;}
    ));

    // Fork options the upstream merge 279b4bfaf removed from the parser (fields + plumbing survived). Generated
    // from the pre-merge parser's own handler bodies -- see the file header.
    // Row S69-UPSTREAM-MERGE-DROPPED-166-FORK-SERVER-CLI-FLAGS-2026-09-27-01.
#include "arg-fork-restored.inc"

    return ctx_arg;
}

void common_params_add_preset_options(std::vector<common_arg> & args) {
    // arguments below won't be treated as CLI args, only preset options
    args.push_back(common_arg(
        {"load-on-startup"}, "NAME",
        "in server router mode, autoload this model on startup",
        [](common_params &, const std::string &) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_LOAD_ON_STARTUP).set_preset_only());

    args.push_back(common_arg(
        {"stop-timeout"}, "SECONDS",
        "in server router mode, force-kill model instance after this many seconds of graceful shutdown",
        [](common_params &, int) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_STOP_TIMEOUT).set_preset_only());

    args.push_back(common_arg(
        {"dedup-cache-models"}, "0|1",
        "in server router mode, hide a cached model from the model list when this preset resolves to the same model file",
        [](common_params &, const std::string &) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_DEDUP_CACHE_MODELS).set_preset_only());

    // args.push_back(common_arg(
    //     {"pin"},
    //     "in server router mode, do not unload this model if models_max is exceeded",
    //     [](common_params &) { /* unused */ }
    // ).set_preset_only());
}
