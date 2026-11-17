//
// Copyright (C) 2023-2025 The llama.cpp authors
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

// Various helper functions and utilities

#pragma once

#include "llama-cpp.h"

#include "ggml-opt.h"
#include "ggml.h"
#include "ggml-cpu.h"   // CPU-side declarations (ggml_graph_plan/compute, ggml_cpu_has_*, ggml_get/set_*) moved here, per upstream
#include "llama.h"

#include "sampling.h"

#define LOG_NO_FILE_LINE_FUNCTION
#include "log.h"
#include "disk-tier.h"   // common_disk_blob: the tier reference a context checkpoint holds (--cache-disk-dir)
#include <memory>
#include <array>
#include <list>
#include <set>
#include <cmath>
#include <string>
#include <sstream>
#include <string_view>
#include <vector>
#include <random>
#include <thread>
#include <unordered_map>
#include <tuple>
#include <map>
#include <sstream>
#include <variant>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstdio>

struct llama_cram_probe_result; // src/llama-cram-probe.h
class llama_cram_soft_store;    // src/llama-cram-soft.h

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif // _WIN32

#define COM_DBG(fmt, ...) LOG_DBG("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_TRC(fmt, ...) LOG_TRC("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_INF(fmt, ...) LOG_INF("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_WRN(fmt, ...) LOG_WRN("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_ERR(fmt, ...) LOG_ERR("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

struct common_time_meas {
    common_time_meas(int64_t & t_acc, bool disable = false);
    ~common_time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

struct llama_lora_adapter_info {
    std::string path;
    float scale;
};

struct llama_lora_adapter_container : llama_lora_adapter_info {
    struct llama_lora_adapter * adapter;
};

using llama_tokens = std::vector<llama_token>;

// build info
extern int LLAMA_BUILD_NUMBER;
extern char const * LLAMA_COMMIT;
extern char const * LLAMA_COMPILER;
extern char const * LLAMA_BUILD_TARGET;

struct common_adapter_lora_info {
    std::string path;
    float scale;

    std::string task_name;
    std::string prompt_prefix;

    struct llama_adapter_lora * ptr;
};

// Sum of the tensor data bytes of every adapter GGUF (header read only, no data loaded). The scale is ignored: a
// scale-0 adapter is still resident. A file that cannot be read adds 0 and logs a warning.
size_t common_lora_adapters_tensor_bytes(const std::vector<common_adapter_lora_info> & adapters);
// Bytes to add to every device's fit target so --lora weights do not get the memory the fit hands to layers/context.
// 0 when nothing is offloaded to a GPU (CPU-only).
size_t common_lora_fit_reserve_bytes(const std::vector<common_adapter_lora_info> & adapters, bool gpu_offload);

using llama_tokens = std::vector<llama_token>;

// RESTORED 2026-09-13, verbatim from this repository's own history: commit 5cdb27e09
// "finetune: SGD optimizer, more CLI args (#13873)", an ancestor of HEAD. `git log -S "struct
// lr_opt" -- common/common.h` returns exactly that one commit.
//
// The merge lost the STRUCT while keeping everything around it: common_params still declares
// `struct lr_opt lr;` (below), and common.cpp still DEFINES its methods -- lr_opt::init() at
// common.cpp:2843 and lr_opt::get_lr(float) at :2856, which match the two declared here exactly.
// So the only missing piece was the definition, and every translation unit including common.h
// failed with `field 'lr' has incomplete type 'lr_opt'`.
//
// Placed here rather than beside its use because it must precede common_params and depends on
// nothing but float and unsigned.
struct lr_opt {
    float    lr0          = 1e-5; // learning rate at first epoch
    float    lr_min       = -1;
    float    decay_epochs = -1;   // if >0, the learning rate starts at lr0 and decays to lr_min after this many epochs
    float    scale_epoch  = 0;
    float    wd           = 0;
    unsigned epochs       = 2;

    unsigned epoch; // set by optimizer outer (epochs) loop
    // learning rate decay - constant LR per epoch only for now
    float get_lr(float e) const;
    float get_lr() const { return get_lr(epoch); }
    // must call after arg parse, before get_lr
    void init();
};

struct common_control_vector_load_info;

//
// CPU utils
//

struct common_cpu_params {
    int      n_threads                   = -1;
    bool     cpumask[GGML_MAX_N_THREADS] = {false}; // CPU affinity mask.
    bool     mask_valid                  = false;   // Default: any CPU
    enum ggml_sched_priority  priority   = GGML_SCHED_PRIO_NORMAL;  // Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
    bool     strict_cpu                  = false;   // Use strict CPU placement
    bool     flush_denormals             = false;   // --cpu-flush-denormals: FTZ/DAZ in CPU compute workers (opt-in)
    uint32_t poll                        = 50;      // Polling (busywait) level (0 - no polling, 100 - mostly polling)
    bool     place_physical              = false;   // --cpu-placement physical: one worker per physical core (see below)
};

int32_t common_cpu_get_num_physical_cores();
int32_t common_cpu_get_num_math();

// --cpu-placement physical (row S69-DECL-OVERTURN-CPU-AFFINITY-AUTO-2026-10-17-01; behaviour of Thireus/ik_llama #2565's
// automatic mode, generalised from "hybrid P-cores" to "every physical core"). Opt-in; the default placement is unchanged.
// The logical CPUs of each physical core, as global CPU ids in the numbering the ggml threadpool cpumask uses (Windows:
// processor-group base + bit, group bases from GetActiveProcessorCount; Linux: the kernel's cpu number). Windows reads
// GetLogicalProcessorInformationEx(RelationProcessorCore), Linux /sys/devices/system/cpu/cpuN/topology/thread_siblings_list.
// Empty when the topology cannot be read (other OSes, errors).
std::vector<std::vector<int32_t>> common_cpu_core_siblings();
// One logical CPU per core -- the lowest id of each core -- sorted ascending; SMT siblings are left out. Pure function.
std::vector<int32_t> common_cpu_first_cpu_per_core(const std::vector<std::vector<int32_t>> & cores);
// Apply physical placement to one role's params: when place_physical is set and NO explicit mask was given (-C/-Cr and the
// batch/draft variants win), set cpumask to `cpus` and strict_cpu = true so worker i is pinned to the i-th physical core.
// Returns true when the mask was written.
bool common_cpu_apply_physical_placement(common_cpu_params & cpuparams, const std::vector<int32_t> & cpus);
// "physical" -> true, "none" -> false; anything else is rejected.
bool common_cpu_placement_parse(const std::string & value, bool & physical);

// RESTORED 2026-09-13. common.cpp:620 does `params.model.path = DEFAULT_MODEL_PATH;` and the macro was
// defined nowhere, so that translation unit could not compile.
//
// PROVENANCE, from this repository's own history: `git log -S DEFAULT_MODEL_PATH -- common/common.h`
// returns two commits. 8843a98c2 "Improve usability of --model-url & related flags (#6930)" ADDED
//     #define DEFAULT_MODEL_PATH "models/7B/ggml-model-f16.gguf"
// and ec18edfcb "server: introduce API for serving / loading / unloading multiple models (#17470)"
// REMOVED it -- the same upstream change that replaced the flat `std::string model` with a nested
// `struct common_params_model model`.
//
// This fork deliberately did NOT adopt that struct. The decision is recorded in common_params
// below: "NOT adopted: theirs' nested `struct common_params_model model;` ... ours' flat
// model/model_alias(std::string)/model_url/hf_token/hf_repo/hf_file stand." Keeping the flat field
// and keeping the macro that goes with it is the consistent position; the macro only disappeared
// because upstream no longer had a string to assign.
#define DEFAULT_MODEL_PATH "models/7B/ggml-model-f16.gguf"

// RESTORED 2026-09-13, verbatim from this repository's own history: commit ab1401982 "Support
// diffusion models: Add Dream 7B (#14644)", the only revision of common/common.h carrying it.
// common_params needs it for the `diffusion` member added below, which arg.cpp references.
struct common_params_diffusion {
    int32_t steps         = 128;
    int32_t blocks        = 1;        // max block-autoregressive denoising blocks (block-diffusion models)
    bool    visual_mode   = false;    // show progressive diffusion on screen
    bool    visual_progress = false;  // show the step progress bar in visual mode (default: hidden)
    int32_t visual_interval = 1;      // redraw the visual canvas every Nth step (all steps still computed)

    float   eps           = 0;        // epsilon for timesteps
    int32_t block_length  = 0;        // block length for generation

    int32_t algorithm     = 4;        // default algorithm: low-confidence (0=ORIGIN 1=ENTROPY 2=MARGIN 3=RANDOM 4=CONFIDENCE)
    float   alg_temp      = 0.0f;     // algorithm temperature

    float   cfg_scale     = 0;        // classifier-free guidance scale
    bool    add_gumbel_noise = false; // add gumbel noise to the logits if temp > 0.0

    // entropy-bound decoder (DiffusionGemma canvas models); params default to GGUF metadata, then reference
    int32_t eb_mode          = 0;     // 0=auto (on for canvas models), 1=force on, 2=off
    float   eb_t_min         = -1.0f; // <0 / <=0 -> not overridden on the command line
    float   eb_t_max         = -1.0f;
    float   eb_entropy_bound = -1.0f;
    int32_t eb_stability     = -1;
    float   eb_confidence    = -1.0f;
    int32_t eb_max_steps     = -1;
    int32_t eb_kv_cache      = 0;     // prefix KV cache: 0=auto (on for single-GPU canvas), 1=on, 2=off
    int32_t eb_gpu_sampling  = 0;     // device-resident SC: 0=auto (on for single-GPU canvas), 1=on, 2=off
    int32_t eb_gpu_sample_reduce = 0; // Stage-1 device argmax/entropy/sample reduction: 0=auto (on for single-GPU), 1=on, 2=off
};

// P-cores, one thread per physical core (no E-cores, no SMT siblings)
std::vector<int32_t> cpu_get_math_cpus();

// E-cores, one per physical core (Intel hybrid)
std::vector<int32_t> cpu_get_efficiency_cpus();

// explicit affinity, else auto-detected math CPUs
std::vector<int32_t> cpu_affinity_resolve(const std::vector<int32_t> & cpus, bool auto_detect);

// explicit affinity, else auto-detected E-cores
std::vector<int32_t> cpu_affinity_resolve_draft(const std::vector<int32_t> & cpus, bool auto_detect);

bool cpu_affinity_parse_mask (const std::string & value, std::vector<int32_t> & cpus);
bool cpu_affinity_parse_range(const std::string & value, std::vector<int32_t> & cpus);

enum llama_example {
    LLAMA_EXAMPLE_COMMON,
    LLAMA_EXAMPLE_SPECULATIVE,
    LLAMA_EXAMPLE_MAIN,
    LLAMA_EXAMPLE_EMBEDDING,
    LLAMA_EXAMPLE_PERPLEXITY,
    LLAMA_EXAMPLE_RETRIEVAL,
    LLAMA_EXAMPLE_PASSKEY,
    LLAMA_EXAMPLE_IMATRIX,
    LLAMA_EXAMPLE_BENCH,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CVECTOR_GENERATOR,
    LLAMA_EXAMPLE_EXPORT_LORA,
    LLAMA_EXAMPLE_MTMD,
    LLAMA_EXAMPLE_LOOKUP,
    LLAMA_EXAMPLE_PARALLEL,
    LLAMA_EXAMPLE_TTS,
    LLAMA_EXAMPLE_DIFFUSION,
    LLAMA_EXAMPLE_FINETUNE,

    // ADDED 2026-09-13. These ten are referenced across common/ -- overwhelmingly by arg.cpp's
    // set_examples({...}) calls -- and were absent from this enum, which accounted for 218 of the
    // 265 remaining errors in the `common` target. The set was computed rather than guessed:
    // every LLAMA_EXAMPLE_* token used in arg.cpp, arg.h, common.cpp and preset.cpp, minus every
    // token declared here.
    //
    // LLAMA_EXAMPLE_CLI is upstream's rename of this fork's LLAMA_EXAMPLE_MAIN. MAIN is kept above
    // because it is this enum's original spelling and removing it is a separate decision from
    // making the tree compile; the two are distinct enumerators and nothing here depends on them
    // being equal. common.h.pre-d5 carries BOTH enums -- one with MAIN at :137 and one with CLI at
    // :424 -- which is the merge having seen both and kept the wrong one.
    //
    // Appended in a block rather than interleaved, because these are ordinary enumerators with no
    // explicit values and any reordering would change every one of them.
    LLAMA_EXAMPLE_CLI,
    LLAMA_EXAMPLE_COMPLETION,
    LLAMA_EXAMPLE_BATCHED,
    LLAMA_EXAMPLE_DEBUG,
    LLAMA_EXAMPLE_DOWNLOAD,
    LLAMA_EXAMPLE_EXPORT_GRAPH_OPS,
    LLAMA_EXAMPLE_FIT_PARAMS,
    LLAMA_EXAMPLE_LOOKAHEAD,
    LLAMA_EXAMPLE_RESULTS,
    LLAMA_EXAMPLE_TOKENIZE,

    // ADDED 2026-09-26: this fork's own llama-sweep-bench. Its six options (-nrep, --sweep-stride,
    // --sweep-memory, -wb, --output-format md|jsonl, --minilog) lived in the legacy gpt_params_parse
    // (ik common.cpp; --sweep-stride/--sweep-memory from 1ce4bb973), which the merge replaced with
    // arg.cpp -- the fields survived in common_params, their parsers did not. Appended, not
    // interleaved, for the same reason as the block above.
    LLAMA_EXAMPLE_SWEEP_BENCH,

    // MUST STAY LAST: this is the sentinel, and code sizes arrays with it.
    LLAMA_EXAMPLE_COUNT,
};

//
// Common params
//

// dimensionality reduction methods, used by cvector-generator
enum dimre_method {
    DIMRE_METHOD_PCA,
    DIMRE_METHOD_MEAN,
};

// reasoning API response format (not to be confused as chat template's reasoning format)
enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_AUTO,
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY, // Extract thinking tag contents and return as `message.reasoning_content`, or leave inline in <think> tags in stream mode
    COMMON_REASONING_FORMAT_DEEPSEEK,        // Extract thinking tag contents and return as `message.reasoning_content`, including in streaming deltas.
};

enum common_webui {
    COMMON_WEBUI_NONE,
    COMMON_WEBUI_AUTO,
    COMMON_WEBUI_LLAMACPP,
};

enum common_checkpoint_eviction {
    COMMON_CHECKPOINT_EVICTION_AUTO,
    COMMON_CHECKPOINT_EVICTION_FIFO,
    COMMON_CHECKPOINT_EVICTION_VARIANCE
};

common_webui common_webui_from_name(const std::string& format);

common_checkpoint_eviction common_checkpoint_eviction_from_name(const std::string & format);


struct thinking_tokens {
    bool exclude = true;
    std::string begin = "<think>";
    std::string end = "</think>";
};

thinking_tokens thinking_tokens_from_string(const std::string& format);

enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,          // no speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT,         // draft model
    COMMON_SPECULATIVE_TYPE_DFLASH,        // DFlash draft model
    COMMON_SPECULATIVE_TYPE_DSPARK,
    COMMON_SPECULATIVE_TYPE_MTP,           // MTP model
    COMMON_SPECULATIVE_TYPE_EAGLE3,        // eagle draft model
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,  // simple self-speculative decoding
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,   // self-speculative decoding with n-gram keys only
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, // self-speculative decoding with n-gram keys and 4 m-gram values
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,   // self-speculative decoding with 3-level n-gram cache
    COMMON_SPECULATIVE_TYPE_SUFFIX,        // self-speculative suffix-decoding (arXiv:2411.04975)
    COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON, // longest-suffix-match self-speculation over a suffix automaton (O(n) memory, no depth cap)
    COMMON_SPECULATIVE_TYPE_COUNT         // number of types, unknown type
};

std::string common_speculative_type_name_str();
enum common_speculative_type common_speculative_type_from_name(const std::string & name);
std::string common_speculative_type_to_str(enum common_speculative_type type);
bool common_speculative_type_is_dflash_family(enum common_speculative_type type);
bool common_speculative_type_uses_target_features(enum common_speculative_type type);
// true when the stage honours --spec-draft-sample (draws from q and records it): the draft-model and MTP stages. Every other type
// (ngram_*, suffix*, eagle3, dflash, dspark) drafts deterministically and ignores the flag (row ...DRAFT-SAMPLE-EQUIVALENCE-TEST C1(b)).
bool common_speculative_type_draws_from_q(enum common_speculative_type type);
bool common_speculative_type_is_self_spec(enum common_speculative_type type);

struct common_speculative_stage_params {
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;

    int32_t n_max = -1;
    int32_t n_min = -1;
    float   p_min = -1.0f;
    int32_t mtp_heads = -1;

    uint16_t ngram_size_n = 0;
    uint16_t ngram_size_m = 0;
    uint16_t ngram_min_hits = 0;

    int32_t suffix_min_match_len = -1;
    int32_t suffix_max_depth = -1;
    int64_t suffix_max_nodes = -1;
    std::string suffix_corpus;

    bool has_n_max_override() const { return n_max >= 0; }
    bool has_n_min_override() const { return n_min >= 0; }
    bool has_p_min_override() const { return p_min >= 0.0f; }
    bool has_mtp_heads_override() const { return mtp_heads >= 0; }
    bool has_ngram_size_n_override() const { return ngram_size_n > 0; }
    bool has_ngram_size_m_override() const { return ngram_size_m > 0; }
    bool has_ngram_min_hits_override() const { return ngram_min_hits > 0; }
    bool has_suffix_min_match_len_override() const { return suffix_min_match_len >= 0; }
    bool has_suffix_max_depth_override() const { return suffix_max_depth >= 0; }
    bool has_suffix_max_nodes_override() const { return suffix_max_nodes >= 0; }
    bool has_suffix_corpus_override() const { return !suffix_corpus.empty(); }
};

struct common_params_model {
    std::string path        = ""; // model local path                                       // NOLINT
    std::string url         = ""; // model url to download                                  // NOLINT
    std::string hf_repo     = ""; // HF repo                                                // NOLINT
    std::string hf_file     = ""; // HF file                                                // NOLINT
    std::string docker_repo = ""; // Docker repo                                            // NOLINT
    std::string hf_revision = ""; // HF branch, tag or 40-hex commit to pin hf_repo to ("" = main) // NOLINT

    // --- ported from theirs (block 3 union, no name clash) ---
    std::string get_name() const {
        if (!hf_repo.empty()) {
            return hf_repo;
        }
        if (!docker_repo.empty()) {
            return docker_repo;
        }
        return path;
    }

    bool empty() const {
        return get_name().empty();
    }
};

struct common_ngram_mod;

// upstream's draft-model settings (llama.cpp common_params_speculative_draft). Since 2026-11-05 it is the type of
// common_params_speculative::draft below: upstream's spelling, folded into the flat fields by common_params_speculative_fold_draft.
struct common_params_speculative_draft {
    int32_t n_max = 3;
    int32_t n_min = 0;

    float p_split = 0.1f;
    float p_min   = 0.0f;

    bool backend_sampling = true;

    bool probabilistic = false; // sample the draft and verify by rejection, instead of argmax and match

    common_params_model mparams;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    int32_t n_gpu_layers = -1;

    int32_t n_ubatch = 0; // physical batch size of the draft context (0 - use the target's n_ubatch)

    ggml_type cache_type_k = GGML_TYPE_F16;
    ggml_type cache_type_v = GGML_TYPE_F16;

    enum llama_lazy_mode lazy_mode = LLAMA_LAZY_MODE_AUTO; // on-demand reading of tensors marked by the arch

    common_cpu_params cpuparams;
    common_cpu_params cpuparams_batch;

    std::vector<ggml_backend_dev_t> devices;

    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
};

struct common_params_speculative {
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE; // type of speculative decoding

    // Generic speculative checkpoint mode, recurrent spelling is a compatibility alias.
    int spec_ckpt_mode = LLAMA_SPEC_CKPT_AUTO;

    std::string devices;
    std::string params;
    int32_t n_threads = -1;
    int32_t n_threads_batch = -1;

    int32_t n_max = 16; // number of tokens to draft during speculative decoding
    int32_t n_min = 0; // minimum number of tokens to draft during speculative decoding
    std::vector<common_speculative_stage_params> stages; // explicit stage chain for single-spec or self-spec + model fallback
    int32_t mtp_heads = 1; // MTP heads to use; 1 is the default, while >1 and 0 (all model heads) are experimental

    // synthetic acceptance (upstream): a fixed draft length / per-position acceptance rates instead of a real drafter
    double synth_len = -1.0;
    std::vector<double> synth_rates;

    // Samplers for DFlash2
    float    draft_temperature = 0.0f;
    uint32_t draft_seed = LLAMA_DEFAULT_SEED;

    float   p_split = 0.1f; // speculative decoding split probability
    float   p_min = 0.75f; // minimum speculative decoding probability (greedy)
    // true once --spec-draft-p-min / LLAMA_ARG_SPEC_DRAFT_P_MIN / a stage override sets p_min. The DRAFT-MODEL gates
    // (common/speculative.cpp) apply p_min only then, so their default is upstream's 0.0 (engines/llama.cpp
    // common/common.h:331) -- the behaviour they actually had until 2026-09-28, when the gate read an unnormalised
    // probability that was always 1.0 and never fired. MTP keeps reading p_min (0.75) as before: its gate was live.
    bool    p_min_explicit = false;
    // --spec-draft-sample: the draft model SAMPLES its token from its own distribution q (instead of taking the
    // argmax) and records q, and the server verifies with rejection sampling (common/spec-accept.h). With an argmax
    // draft, q is one-hot and rejection sampling equals sample-and-match, so the switch only matters with it on.
    bool    draft_sample = false;
    // --spec-draft-grammar: the draft model drafts under a copy of the request's grammar (or llguidance constraint),
    // so it never proposes a token the target's grammar forbids. Draft-model impl only. With --spec-draft-sample the draft is
    // drawn with the grammar applied first and q is the masked, renormalised distribution (grammar-mask renormalisation, row
    // S69-ENGREV-UPSTREAM-MERGE-DRAFT-SAMPLE-EQUIVALENCE-TEST-2026-11-04-01). Row S69-LMDEPLOY-PICK-GRAMMAR-MASKED-SPEC-DRAFT-2026-10-18-01.
    bool    draft_grammar = false;
    // --spec-draft-validate (default ON): whatever proposer made the draft (n-gram, corpus, suffix, EAGLE, draft model ...), the server
    // cuts it to the longest prefix the request's grammar accepts BEFORE the verify batch is built, so verify width is not spent on
    // tokens the grammar forbids. Lossless: a forbidden token can never be what the grammar-constrained target samples.
    // vLLM structured_output/__init__.py:294-312. Row S69-VLLM-W2-GRAMMAR-VALIDATED-DRAFTS-2026-10-27-01.
    bool    draft_validate = true;
    // --spec-async (default OFF, opt-in): draft || verify overlap for the draft-MODEL drafter. While the target verifies draft k, a worker
    // thread re-evaluates [id_last, draft k] on the draft context (work the serial path does AFTER verify) and keeps drafting from the
    // all-accepted path (PEARL post-verify, arXiv 2408.11850; AMUSD arXiv 2410.17375). Lossless: the target still decides every token; a
    // partial acceptance rolls the draft context back with llama_memory_seq_rm exactly as the serial path does. Design + measurements:
    // temp/research/findings/S69_ASYM_SPEC_PIPELINES_AND_KV_COMPRESSION_2026-11-06.md "Built 2026-11-06". Row S69-ASYM-SPEC-ASYNC-DRAFT-VERIFY-OVERLAP-PEARL-AMUSD-2026-11-06-01.
    bool    async = false;
    // --spec-async-wait-ms: how long the main thread lets the lookahead keep running after the target verify finished (0 = stop it at the next
    // token boundary; the mandatory first step, the [id_last, draft] re-evaluation, always completes)
    int32_t async_wait_ms = 0;
    // --spec-async-informed-join (default OFF, opt-in; row S69-SPECASYNC-INFORMED-JOIN-2026-11-06-01): at the join the target logits of the
    // verify rows are already on the host, so for GREEDY requests the join knows whether the lookahead will be used: the whole draft was
    // accepted and the target's own token equals the lookahead's c1. Only then does the worker keep drafting (bounded by the measured
    // draft step time and, when > 0, by --spec-async-wait-ms); otherwise it is stopped at once. Non-greedy requests and any row it cannot
    // read keep the wait_ms rule. Lossless either way: it only decides how long the main thread waits.
    bool    async_informed = false;

    // ngram-based speculative decoding

    uint16_t ngram_size_n = 12; // ngram size for lookup
    uint16_t ngram_size_m = 48; // mgram size for speculative tokens
    uint16_t ngram_min_hits = 1; // minimum hits at ngram/mgram lookup for mgram to be proposed

    std::shared_ptr<common_ngram_mod> ngram_mod;

    // suffix-decoding specific
    int32_t     suffix_min_match_len = 5;  // minimum context match length
    int32_t     suffix_max_depth     = 64; // suffix tree maximum depth
    int64_t     suffix_max_nodes     = 1 << 20; // suffix tree node cap per sequence (oldest history is dropped past it; 0 = unbounded; ~150 B per node)
    std::string suffix_corpus;             // path to corpus file for offline pre-warming (.json or .bin)

    std::string lookup_cache_static;  // path of static ngram cache file for lookup decoding           // NOLINT
    std::string lookup_cache_dynamic; // path of dynamic ngram cache file for lookup decoding          // NOLINT

    // draft-model speculative decoding
    struct common_params_model mparams_dft;

    // --spec-self-draft-type TYPE (S69-IPEXLLM-PICK-SELF-SPECULATIVE-DRAFT-2026-09-29-01): when no draft model is given, derive one from the
    // target itself by quantizing the target GGUF to TYPE (cached next to it) and use that as the draft. Empty = off.
    std::string self_draft_type;

    llama_model * model_dft = nullptr; // a llama_model that can be shared by multiple speculative contexts

    // Contexts for the per-sequence DRAFT implementation (common_speculative_init(params, n_seq)), the
    // fork's flat counterpart of upstream's params.speculative.draft.ctx_tgt / .ctx_dft. Set by the caller
    // (llama-server) once it has created the draft context; NOT owned here. nullptr -> no draft impl.
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    llama_context_params cparams_dft; // these are the parameters for the draft llama_context

    // Owned storage for every pointer common_context_params_to_llama writes into cparams_dft. Both
    // writers (common_speculative_load_draft_model, common_speculative_prepare_mtp_runtime) build
    // cparams_dft from a LOCAL gpt_params copy, and the draft/MTP llama_context is created later in
    // common_speculative_try_init -- so without this the pointers dangle by then. Upstream fixed the
    // cuda_params + offload_policy half (ikawrakow/ik_llama.cpp#2715, 804ee29c: '-cuda graphs=0'
    // was ignored by the MTP context, its CUDA graphs were never disabled and VRAM grew per
    // request, #2684); the fork adds six const char* fields with the same lifetime bug.
    // Rebound by common_speculative_own_cparams_dft (speculative.cpp).
    struct cparams_dft_storage {
        std::string cuda_params;
        std::vector<std::pair<int,int>> offload_policy;
        std::string tri_stats_path;
        std::string cats_thresholds_file;
        std::string moe_stats_path;
        std::string moe_stats_mode;
        std::string moe_placement_strategy;
        std::string moe_placement_stats;
        std::string teal_thresholds_file;
    } cparams_dft_store;

    // draft affinity from --draft-params; empty = inherit the target
    std::vector<int32_t> cpu_affinity;
    bool cpu_affinity_configured = false; // a draft affinity was requested

    int32_t n_ctx = 0;  // draft context size
    int32_t n_ubatch = 0; // --spec-draft-n-ubatch: physical batch size of the draft context (0 - use the target's n_ubatch)
    int32_t n_gpu_layers = -1; // number of layers to store in VRAM for the draft model (-1 - use default)

    std::string model = ""; // draft model for speculative decoding
    std::vector<std::pair<std::string, std::string>> replacements; // main to speculative model replacements
    std::string cache_type_k = ""; // KV cache data type for K for the draft model
    std::string cache_type_v = ""; // KV cache data type for V for the draft model

    // upstream spelling (llama.cpp params.speculative.draft.*). INPUT only: common_params_speculative_fold_draft copies every field that
    // differs from its default into the flat fields of this struct -- which is what everything in this tree reads -- and resets it. It
    // runs in common_params_parse and on the copy in common_base_params_to_speculative. lazy_mode has no flat field and is kept here.
    common_params_speculative_draft draft;

    // ADDED (S69-ARGCPP-SPEC-SHAPE-MISMATCH-2026-09-08-01): arg.cpp's --spec-draft-cpu-mask/
    // -range/-strict/-prio/-poll(-batch) flags need somewhere real to write draft-thread
    // affinity/priority -- flat here (not nested under a `.draft` sub-struct) to match every
    // other field in this struct and stay consistent with n_threads/n_threads_batch above,
    // which common_base_params_to_speculative (speculative.cpp) already reads flat.
    common_cpu_params cpuparams;
    common_cpu_params cpuparams_batch;

    // ADDED, same row: per-draft-model tensor buffer-type overrides (--spec-draft-override-tensor
    // / --spec-draft-cpu-moe family). Populated by arg.cpp; not yet folded into the draft
    // model's llama_model_params by common_model_params_to_llama (BENCH-PENDING, flagged in the
    // progress note -- field now exists so arg.cpp compiles and the value is retained/inspectable,
    // wiring it into the actual draft-model load is a follow-up, not silently invented as done).
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;

    // ADDED, same row: --spec-draft-backend-sampling toggle. Default true, matching
    // common_params_speculative_draft::backend_sampling (upstream's spelling, folded in by common_params_speculative_fold_draft).
    bool backend_sampling = true;

    bool autotune = false; // automatically optimize speculative params for max tokens/sec

    // Choose per round between a leading self-speculative (lookup) stage and the model stage after it, from learned lookup
    // acceptance and measured round cost (common/draft-policy.h). Off: a non-empty lookup draft always wins, as before.
    // BENCH-PENDING: no tok/s A/B yet (needs an MTP-capable GGUF; row S69-STRATA-S2-DRAFT-POLICY-COST-MODEL-MTP-LOOKUP-CHAIN-2026-10-07-01).
    bool draft_policy = false;

    // The draft model path. Two fields name it and the merge split producers from consumers:
    // `-md/--model-draft` and all 8 HF sidecar resolvers in arg.cpp (MTP / DFlash / Eagle3 /
    // DSpark, plan and plan_spec) write mparams_dft.path, while has_dft(), load_draft_model,
    // common_base_params_to_speculative and the server's draft path read the flat `model` -- which
    // NOTHING in the tree writes. So `-md` loaded no draft anywhere: the server skipped speculation
    // silently and llama-spec-bench refused "draft speculative stage requires a draft model"
    // (MEASURED 2026-09-26). Read the path through here, never through either field alone.
    const std::string & dft_path() const {
        return !model.empty() ? model : mparams_dft.path;
    }

    bool has_synth() const {
        return synth_len != -1.0 || !synth_rates.empty();
    }

    bool has_dft() const {
        // ik c75ad6b6b (#2588): an external draft model is present only when a draft PATH is set;
        // --draft-params alone now carries context/affinity parameters for embedded stages
        // (common_speculative_prepare_startup reads `!has_dft() && !params.empty()` as that case).
        // dft_path() keeps this fork's two-field read (see above).
        return !dft_path().empty();
    }

    std::vector<common_speculative_stage_params> get_resolved_stages() const;
    common_params_speculative with_stage_overrides(const common_speculative_stage_params & stage) const;
    bool has_stage_chain() const;
    bool has_stage_type(common_speculative_type stage_type) const;
    bool has_dflash_family_stage() const;
    bool uses_target_features() const;
    void remove_stage_type(common_speculative_type stage_type);
    bool has_composite_stage_chain() const;
    bool needs_dft_model() const;
    void clear_dft();
    int32_t get_max_stage_n_max() const;
    int32_t get_min_usable_stage_n_min() const;

};

bool common_speculative_validate_chain(const common_params_speculative & params, std::string * error = nullptr);
// folds upstream's `draft` sub-struct into the flat fields (see common_params_speculative::draft)
void common_params_speculative_fold_draft(common_params_speculative & sp);
std::string common_speculative_stage_chain_to_str(const common_params_speculative & params);

// MOVED OUT OF struct common_params 2026-09-13. These three type definitions were pasted
// INSIDE the struct body (they sat at lines 531-576, between `sweep_bench` and the sampling
// members), which made every one of their names member-scoped: arg.cpp:1956 referring to a
// bare COMMON_CONVERSATION_MODE_ENABLED could not see it, because its real name was
// common_params::COMMON_CONVERSATION_MODE_ENABLED.
//
// Nothing about the definitions changes -- they are moved verbatim, only to namespace scope,
// ahead of the struct that uses them. common_cpu_params (:131) and common_params_model (:329)
// are both declared above this point, so the two structs below still have what they need.
//
// Same shape as the inventory defect recorded earlier this session, read from the other side:
// PRESENCE INSIDE A BRACE RANGE IS NOT MEMBERSHIP -- and a DEFINITION inside the range
// silently BECOMES a member.
// --- ported from theirs (block 3 union, additive, no name/type clash) ---
enum common_conversation_mode {
    COMMON_CONVERSATION_MODE_DISABLED = 0,
    COMMON_CONVERSATION_MODE_ENABLED  = 1,
    COMMON_CONVERSATION_MODE_AUTO     = 2,
};


// theirs' ngram-modifier config surface (DIFFERENT NAME from ours' opaque
// `struct common_ngram_mod;` forward-decl above -- possibly the same concept exposed
// differently; NOT verified, kept alongside rather than guessed).
struct common_params_speculative_ngram_mod {
    int32_t n_match = 24;

    int32_t n_max = 64;
    int32_t n_min = 48;
};

struct common_params {
    // Every CLI flag the parser actually SAW, canonicalised ('_' is normalised to '-' inside '--'
    // arguments by the parse loop before this is filled). Used to answer "did the user set this?"
    // when applying model-embedded sampling defaults, because comparing a parsed value against the
    // hardcoded default cannot answer it -- a user who passes the default value looks identical to
    // one who passed nothing at all.
    std::set<std::string> args_seen;

    std::string devices;
    uint32_t seed                 = LLAMA_DEFAULT_SEED; // RNG seed

    // common_cpu_get_num_math, not cpu_get_num_math: upstream prefixed these helpers and this call
    // site kept the old spelling, which the compiler named exactly ("did you mean
    // 'common_cpu_get_num_math'?"). The declaration is 253 lines above this at common.h:113.
    int32_t n_threads             = common_cpu_get_num_math();
    int32_t n_threads_batch       =      -1; // number of threads to use for batch processing (-1 = use n_threads)
    std::vector<int32_t> cpu_affinity;       // logical CPU ids to pin worker threads to (empty = no explicit pinning)
    bool    cpu_affinity_auto     = false;   // pin to hybrid P-cores when cpu_affinity is empty (--cpu-affinity)
    bool    cpu_affinity_configured = false; // a CPU affinity option was given
    int32_t n_predict             =      -1; // new tokens to predict
    int32_t n_ctx                 =       0; // context size
    int32_t n_batch               =    2048; // logical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_ubatch              =     512; // physical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep                =       0; // number of tokens to keep from initial prompt
    int32_t n_chunks              =      -1; // max number of chunks to process (-1 = unlimited)
    int32_t n_parallel            =       1; // number of parallel sequences to decode

    // Classifier-Free Guidance (row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01). Extra
    // llama seq ids reserved ON TOP OF n_parallel (common_context_params_to_llama /
    // common_model_params_to_llama add this to n_seq_max) so a per-slot guidance sequence can share
    // the slot's own context/KV instead of needing a second llama_context. Default 0: n_seq_max is
    // then exactly n_parallel, byte-identical to a build that predates this field -- a server that
    // never sets --cfg-guidance-slots pays no extra memory. NOTE (found by reading
    // src/llama-model.cpp before adding this): n_seq_max is not an isolated seq-id-capacity knob --
    // llama-model.cpp:3301 computes n_ctx_per_seq = cparams.n_ctx / cparams.n_seq_max for YaRN
    // long/short rope-factor selection, and cache_size() sizes recurrent/hybrid per-stream state
    // buffers from n_seq_max. Both are PRE-EXISTING couplings that --parallel already has today
    // (any n_parallel increase already shrinks n_ctx_per_seq the same way); this field extends an
    // accepted existing behaviour rather than introducing a new one, but a --cfg-guidance-slots N
    // server operator should know slot-level n_ctx_per_seq shrinks by the same rule --parallel does.
    int32_t n_cfg_guidance_slots  =       0;

    // --decision-seqs: extra llama seq ids reserved for llama-server's /v1/decision endpoint (tools/parallel-decision), on top of
    // n_parallel + n_cfg_guidance_slots + the radix cache's ids (the decision ids are the LAST range). 0 (default) = endpoint disabled and
    // n_seq_max is byte-identical to a build without the flag. Min 3 when set: cached prefix, one trunk, one branch.
    int32_t n_seq_decision        =       0;

    // ADOPTED 2026-09-13, defaults RECOVERED from this repository's own history rather than
    // chosen: commit dd1ea5243 "llama : support multi-output backend sampling (#25532)" is the
    // only revision of common/common.h carrying n_outputs_max (git log -S, --all), and these two
    // lines are its :449-450 verbatim.
    //
    // ADOPT, not adapt: no member of this struct served the purpose under another name, and the
    // capability is LIVE in this fork's engine rather than dormant -- llama_context_params
    // declares both (include/llama.h:595-596) and src/llama-context.cpp reads them at :299-301,
    // :878, :1095, :1505, :1972, :2499, :2646-2653 and :2747. The call site that needed them,
    // speculative.cpp:1606-1616, widens the output budget for the dflash/dspark family, which
    // samples every position of a decoded noise block instead of just the last one.
    //
    // They are wired through common_context_params_to_llama in common.cpp; a member here that
    // nothing forwards would be a setting that silently does nothing.
    int32_t n_outputs_max         =     0; // max outputs in a batch (0 = n_batch)
    int32_t n_outputs_max_per_seq =     1; // max outputs per sequence
    int32_t n_sequences           =       1; // number of sequences to decode
    float   p_split               =    0.1f; // speculative decoding split probability
    int32_t n_gpu_layers          =      -1; // number of layers to store in VRAM (-1 - use default)
    int32_t main_gpu              =       0; // the GPU that is used for scratch and small tensors
    int32_t max_gpu               =       0; // max number of GPUs to use at a time for split mode "graph"
    int32_t ncmoe                 =       0; // number of layers in which MoE tensors are left in VRAM
    size_t moe_cache_size         =       0; // total GPU cache budget in bytes for CPU-resident MoE experts
    bool   moe_cache_auto         =   false; // --moe-cache-mib auto: --fit chooses moe_cache_size from the free VRAM
    float  moe_cache_cpu_share    =    0.0f; // share of the MoE cache misses computed on the CPU instead of uploaded; < 0 = auto (measured at load)
    int    moe_cache_policy       =       0; // enum llama_moe_cache_policy: 0 slru, 1 classlru
    int32_t fit_margin            =       0; // safety margin for auto-fit in MiB
    bool    fit                   =   false; // automatically fit model (for now just using MoE tensor overrides)
    int32_t worst_graph_tokens    =       0; // number of tokens to use when reserving the worst graph
    float   tensor_split[128]     =     {0}; // how split tensors should be distributed across GPUs
    int32_t grp_attn_n            =       1; // group-attention factor
    int32_t grp_attn_w            =     512; // group-attention width
    // SnapKV/H2O KV-eviction (S69 #22). 0 = off (default). snapkv compresses the prompt KV once at
    // end-of-prefill; h2o evicts during decode keeping heavy-hitters + a recent window.
    int32_t snapkv_cap            =       0; // SnapKV max prompt KV to keep (0=off)
    int32_t snapkv_window         =      64; // SnapKV observation window (recent queries)
    int32_t kvflash_pool_tokens   =       0; // S69 #66 KVFlash paged-KV resident pool in tokens (0=off)
    // S69 #220 offload-attention. Default OFF: it changes WHICH tokens are attended, which
    // is a retrieval-quality change and belongs in an A/B rather than in a default.
    bool    offload_attn          =   false; // enable offload-attention
    int32_t offload_attn_page     =      16; // tokens per KV page (block-table granularity)
    float   offload_attn_budget   =   0.02f; // fraction of pages attended per step (>=1.0 => a count)
    // 0 quest, 1 shadowkv, 2 retattn -- BUT SEE THE NOTE BELOW: as of 2026-08-20 this value
    // reaches cparams and the engagement trace and does NOT change what either graph op
    // computes. MEASURED on served output: selectors 0/1/2 produce byte-identical text on both
    // -fa 1 and -fa 0 (similarity 1.0000, Llama-3.2-1B-Q8_0, temp 0, fixed seed). It selects
    // only in the bridge path (offload_attend_step -> select_pages), which the graph never
    // calls. Tracked: S69-OFFLOAD-ATTN-SELECTOR-INERT-IN-GRAPH-OP-01.
    int32_t offload_attn_selector =       0;
    // Leading KV pages pinned regardless of score. CONSUMED BY BOTH GRAPH OPS as of 2026-08-21
    // (mask_from_kq_op on the FA path, mask_op on the non-FA path) -- NOT only by select_pages,
    // which is exactly what made the FIRST A/B of this flag return byte-identical output at
    // sink=0 and sink=1. Any new offload knob must be READ at both graph ops or it is dormant in
    // serving: reaching cparams and the engagement trace proves nothing about what the op
    // computes. The trace prints sink= for that reason.
    int32_t offload_attn_sink     =       0;
    int32_t offload_attn_recent   =       0; // recent local window: pages ending at the query's page, kept in addition to the top-k (both graph paths)
    // The first N layers attend densely (never masked). HiP's dense_layers (hip_config.py:366-372, default layers 0-3):
    // early layers attend like a sliding window and lose the most from page dropping (arXiv 2502.08910 App. D).
    int32_t offload_attn_dense_layers = 0;
    // Non-empty = a "-N" (last N) or explicit-list spec (e.g. "0-2,45-47"); parsed against n_layer when the context is built. Overrides the count above.
    std::string offload_attn_dense_spec;
    // -1 = not set (env GGML_OFFLOAD_ATTN_HEAD_REDUCE or the built-in default), 0 = union of per-head top-k, 1 = max over heads.
    int32_t offload_attn_head_reduce = -1;
    // FA path: reuse the page selection for R single-token decode steps (HiP mask_refresh_interval.py:9-30, default intervals 8-64).
    int32_t offload_attn_refresh = 1;
    // FA path page scan: 0 exact, 1 hierarchical bisect (HiP coarse-to-fine; approximate, recall measured by offload-attn-recall).
    int32_t offload_attn_scan = 0;
    // Floor on kept KV in TOKENS. See the long note on OffloadConfig::min_kept_tokens: without
    // it, keep collapses to one page at short context and the DEFAULT budget produces word
    // salad. 0 disables the floor and restores the pre-2026-08-21 behaviour.
    int32_t offload_attn_min_tokens = 256;
    int32_t h2o_heavy             =       0; // H2O heavy-hitter budget (0=off)
    int32_t h2o_recent            =     512; // H2O recent window kept
    int32_t n_print               =      -1; // print token count every n tokens (-1 = disabled)
    float   rope_freq_base        =    0.0f; // RoPE base frequency
    float   rope_freq_scale       =    0.0f; // RoPE frequency scaling factor
    float   yarn_ext_factor       =   -1.0f; // YaRN extrapolation mix factor
    float   yarn_attn_factor      =   -1.0f; // YaRN magnitude scaling factor
    float   yarn_beta_fast        =   -1.0f; // YaRN low correction dim
    float   yarn_beta_slow        =   -1.0f; // YaRN high correction dim
    int32_t yarn_orig_ctx         =       0; // YaRN original context length
    float   defrag_thold          =   -1.0f; // KV cache defragmentation threshold
    float   ban_phrases_bias      = -999.0f; // logit bias applied to ban phrases
    int32_t max_extra_alloc_MiB   =     256; // additional VRAM per GPU the scheduler may allocate for more efficient compute graph evaluation
    int32_t nrep                  =       1; // number of repetitions used in sweep bench
    int32_t sweep_stride          =       1;
    bool    sweep_memory          =   false;
    bool    sweep_bench           =   false;


// NOTE: common_speculative_type (theirs' own 11-value enum), common_sampler_type,
// common_grammar_type/common_grammar/common_grammar_trigger*, common_params_sampling_config,
// llama_example, and struct common_params_sampling are INTENTIONALLY NOT unioned here --
// see this script's docstring findings #1 and #5, and decision #5 itself. The grammar/
// sampler-config types already live in sampling.h (ours' side); common_speculative_type and
// llama_example clash by name with ours' existing definitions; common_params_sampling is
// replaced by the sampling.h struct per decision #5.

    ggml_backend_sched_eval_callback cb_eval = nullptr;
    void * cb_eval_user_data                 = nullptr;

    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;

    enum llama_rope_scaling_type rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    enum llama_pooling_type      pooling_type      = LLAMA_POOLING_TYPE_UNSPECIFIED; // pooling type for embeddings
    enum llama_attention_type    attention_type    = LLAMA_ATTENTION_TYPE_UNSPECIFIED; // attention type for embeddings

    // // sampling parameters
    struct common_params_sampling sampling; // renamed from sparams, MEASURED: server-schema.cpp/server-task.cpp already use .sampling. (not UU, staged)
    struct common_params_speculative speculative;

    // NESTED 2026-09-13, reversing the flat-model non-adoption recorded at the bottom of this
    // struct. struct common_params_model is at :329 of this same header, already carries all
    // five fields as .path/.url/.hf_repo/.hf_file/.docker_repo, and is already the type of three
    // other members here (speculative.mparams_dft, speculative_draft.mparams, mmproj). So this
    // was never "adopt a type we do not have" -- only "which spelling does common_params use".
    //
    // COUNTED FIRST (temp/_measure_model_out.txt, both controls passing), across the 644 built
    // sources: 120 flat uses against 62 nested. The raw counts favour flat and they are the wrong
    // reading, because both sides are upstream-derived files that get re-merged. What makes flat
    // expensive is written into arg.cpp itself:
    //   * :417-421 SYNTHESISES a common_params_model out of the flat siblings purely to satisfy
    //     common_download_get_hf_plan(const common_params_model &), with common/download.h noted
    //     there as a forbidden file it may not modify;
    //   * :451-455 and :506-509 keep a SECOND, flat-typed variant of handle_url beside the real
    //     one, because "there is no nested common_params_model object to bind a reference to for
    //     params.model itself";
    //   * :565-566 retypes a helper so both common_params_model::path and the flat params.model
    //     can reach it.
    // Those shims exist only to bridge the two spellings and are re-derived at every merge.
    //
    // hf_token stays flat below: it is a credential, not a model locator, and it is a separate
    // member upstream too.
    struct common_params_model model;            // model path / url / hf repo+file / docker repo
    // WIDENED to a set 2026-09-13, deliberately reversing the "flat model_alias(std::string)
    // stands" half of the non-adoption noted at :683. The rest of that note is untouched --
    // model/model_url/hf_token/hf_repo/hf_file stay flat.
    //
    // MEASURED, not assumed: the maintained server (tools/server, built -- tools/CMakeLists.txt
    // names it) cannot compile against the string form. server-context.cpp:1292 does
    // `*params_base.model_alias.begin()`, :1301 assigns it straight into a
    // `std::set<std::string> model_aliases`, and server.cpp:163 calls `.insert(model_name)`.
    // The only consumers of the string form are examples/server -- NOT in the build,
    // examples/CMakeLists.txt does not name it -- and examples/spec-bench, which only drops the
    // value into a json object and takes an array as happily as a string.
    //
    // Multi-alias is also the strictly larger capability: one name is a set of size one.
    std::set<std::string> model_alias;           // model name aliases (API-visible)
    // model_url / hf_repo / hf_file / docker_repo REMOVED 2026-09-13 -- they are now
    // model.url / model.hf_repo / model.hf_file / model.docker_repo, see the note above.
    // The docker_repo flat sibling added for S69-ARGCPP-MODEL-NESTED-VS-FLAT-2026-09-08-01 goes
    // with them: its own comment said it existed only because "theirs' nested
    // common_params_model::docker_repo was not adopted as a member here", and now it is.
    std::string hf_token             = ""; // HF token
    std::string prompt               = "";
    std::string prompt_file          = ""; // store the external prompt file name
    std::vector<int32_t> prompt_token_ids;  // --token-file: prompt as token ids; non-empty = skip tokenizing the prompt text
    std::string image_calib_dir      = ""; // llama-imatrix --image-calib: directory of images whose vision-token activations are calibrated on (needs --mmproj)
    bool prompt_is_binary            = false; // don't fool around when the prompt contains binary data (as it is for multiple choice)
    std::string path_prompt_cache    = ""; // path to file for saving/loading prompt eval state
    std::string input_prefix         = ""; // string to prefix user inputs with
    std::string input_suffix         = ""; // string to suffix user inputs with
    std::string logdir               = ""; // directory in which to save YAML log files
    std::string lookup_cache_static  = ""; // path of static ngram cache file for lookup decoding
    std::string lookup_cache_dynamic = ""; // path of dynamic ngram cache file for lookup decoding
    std::string logits_file          = ""; // file for saving *all* logits
    std::string rpc_servers          = ""; // comma separated list of RPC servers

    std::string cuda_params          = ""; // comma separated list of cuda parameters key=value1,key2=value2

    std::vector<std::string> in_files;     // all input files
    std::vector<std::string> antiprompt;   // strings upon which more user input is prompted (a.k.a. reverse prompts)
    std::vector<std::string> ban_phrases;  // strings that are banned in generation
    int32_t banned_n                 =  1; // number of tokens that are banned in the phrase
    size_t n_buffer                  =  0; // number of token buffers for string ban
    bool can_ban_phrases             = true;  // whether to ban strings

    std::vector<std::vector<std::tuple<
        uint32_t        // lower codepoint
        ,uint32_t       // upper codepoint
        ,std::string    // unicode script name
        ,float          // bias
    >>> allow_rules;
    std::vector<std::string> allow_pieces;  // each token to allowlist
    std::vector<std::string> allow_kws;     // keywords
    size_t allow_kw_delay = 0;  // minimum n_decoded before first keyword is active (was uninitialised: a
                                // default-initialised common_params read an indeterminate delay)

    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
    std::vector<std::pair<int,int>> offload_policy;
    std::vector<int> fit_margin_array;

    bool lora_init_without_apply = false; // only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_lora_adapter_apply)
    // WIDENED 2026-09-13 from std::vector<llama_lora_adapter_info>. common_adapter_lora_info
    // (common.h:85) is a strict SUPERSET of llama_lora_adapter_info (common.h:68): same
    // `path` and `scale`, plus `task_name`, `prompt_prefix` and `ptr`. So no field is lost
    // and every existing reader of .path/.scale keeps working unchanged.
    //
    // It is the type the rest of the tree already agreed on: common_set_adapter_lora
    // (common.h:1279), tools/server/server-context.cpp:1602 which assigns this very member
    // into a std::vector<common_adapter_lora_info>, all of tools/server/server-common.h,
    // and tools/export-lora. The narrow type survived only here and in examples/imatrix +
    // examples/export-lora -- NEITHER of which is in the build (examples/CMakeLists.txt does
    // not add_subdirectory either one; the maintained copies are under tools/).
    std::vector<common_adapter_lora_info> lora_adapters; // lora adapter path with user defined scale

    // export-lora only (S69-LORAX-PICK-REQUEST-TIME-LORA-MERGE-TIES-DARE-2026-09-29-01): how several adapters are combined before they are added to the base
    // weights. "linear" (default) is the existing behaviour: every adapter's delta is added with its scale. The others combine the DENSE deltas (see lora-merge.h);
    // the per-adapter scale then acts as the merge weight.
    std::string lora_merge_strategy = "linear"; // linear | ties | dare_linear | dare_ties
    float       lora_merge_density  = 1.0f;     // fraction of each delta kept by ties / dare_*, in [0, 1]
    std::string lora_merge_sign     = "total";  // total | frequency (ties, dare_ties)
    int32_t     lora_merge_seed     = 0;        // dare_*

    // --- ported from theirs (block 4 union, additive, no name/type clash) ---
    std::set<std::string> model_tags;      // model tags (informational, not used for routing)
    std::string path_prompts_log_dir = ""; // directory with logged prompts
    std::string logits_output_dir = "data"; // directory for saving logits output files
    bool        save_logits       = false;  // whether to save logits to files
    std::vector<std::string> tensor_filter; // filter tensor names for debug output (regex)
    // NOT adopted: theirs' nested `struct common_params_model model;` and theirs'
    // `std::set<std::string> model_alias` (multi-alias) -- see docstring findings #3/#4.
    // ours' flat model/model_alias(std::string)/model_url/hf_token/hf_repo/hf_file stand.

    // CAPABILITY-GENUINELY-NA: this adds C++ struct members to a CLI parameter block. Nothing here
    // disables a capability -- it restores fields that arg.cpp already tries to write, and the two
    // it does NOT add are ones upstream itself replaced with a field that IS added here.
    //
    // RESEARCH-AFTER-RESULT: the not-adopted set was searched, not assumed. `git log -S use_mlock
    // --all -- common/common.h` returns three revisions and use_mlock is a direct member of
    // common_params in NONE of them; e6dd0e29a "args: refactor mlock/mmap/directio into load-mode
    // (#20834)" is the change that folded them into llama_load_mode. So the capability is not lost,
    // it moved, and load_mode is adopted below. Likewise flash_attn_type: an inventory of this
    // struct found `bool flash_attn = true` already serving it.
    //
    // --- ADOPTED 2026-09-13: eleven members arg.cpp and common.cpp reference and this struct did
    // not have. Each type and default is VERBATIM from this repository's history (53fd974b4 "Sync
    // master with upstream ggml-org/llama.cpp (82bb48500) (#80)", except diffusion's struct from
    // ab1401982), not reconstructed -- a reconstructed default is worse than a compile error
    // because it compiles.
    //
    // ADOPTING RATHER THAN ADAPTING IS A DECISION, and it was made per member rather than in bulk.
    // An inventory checked each name, and each plausible fork spelling of it, against this struct's
    // OWN text with nested struct bodies stripped -- because a member of a nested type is not a
    // member of this one. All eleven came back ABSENT: nothing here already serves these purposes,
    // so adopting them cannot create the two-places-to-write-one-setting problem that the `model`
    // note directly above exists to prevent.
    common_cpu_params cpuparams;
    common_cpu_params cpuparams_batch;

    struct common_params_diffusion   diffusion;

    bool    fit_params         = true;  // whether to fit unset model/context parameters to free device memory
    int32_t fit_params_min_ctx = 4096;  // minimum context size to set when trying to reduce memory use
    bool    fit_params_print   = false; // print the estimated required memory to run the model
    std::vector<size_t> fit_params_target = std::vector<size_t>(llama_max_devices(), 1024 * 1024*1024);

    enum llama_load_mode  load_mode  = LLAMA_LOAD_MODE_MMAP;   // how to load the model
    enum llama_lazy_mode  lazy_mode  = LLAMA_LAZY_MODE_AUTO;   // on-demand reading of tensors marked by the arch (upstream)
    bool no_alloc = false; // only load metadata and simulate memory allocations (upstream)
    enum llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER; // how to split the model across GPUs

    bool ui = true;
    bool ui_mcp_proxy = false;

    std::vector<common_control_vector_load_info> control_vectors; // control vector with user defined scale

    int32_t verbosity                  = 3;  // LOG_LEVEL_INFO
    int32_t control_vector_layer_start = -1; // layer range for control vector
    int32_t control_vector_layer_end   = -1; // layer range for control vector
    bool    offline                    = false;
    int32_t download_connections       = 1;  // >1: parallel range download of large model files (--download-connections)
    int32_t download_chunk_mib         = 0;  // range size in MiB for the parallel download, 0 = 16 (--download-chunk-mib)

    int32_t ppl_stride      = 0;     // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    int32_t ppl_output_type = 0;     // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
                                     //                                       (which is more convenient to use for plotting)
                                     //
    std::string ppl_split_on;        // perplexity: split the input on this exact string; each piece is scored as its own sequence
    bool   ppl_verbose_chunks = false; // perplexity: print each piece's text to stdout before scoring it
    bool   hellaswag      = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
    size_t hellaswag_tasks  = 400;   // number of tasks to use when computing the HellaSwag score

    bool   winogrande       = false; // compute Winogrande score over random tasks from datafile supplied in prompt
    size_t winogrande_tasks = 0;     // number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

    bool   multiple_choice  = false;  // compute TruthfulQA score over random tasks from datafile supplied in prompt
    size_t multiple_choice_tasks = 0; // number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

    bool   kl_divergence    = false; // compute KL divergence

    bool check             = false; // check rather than generate results for llama-results

    bool usage             = false; // print usage
    bool completion        = false; // print source-able completion script
    bool use_color         = false; // use color to distinguish generations and inputs
    bool special           = false; // enable special token output
    bool interactive       = false; // interactive mode
    bool interactive_first = false; // wait for user input immediately
    bool prompt_cache_all  = false; // save user input and generations to prompt cache
    bool prompt_cache_ro   = false; // open the prompt cache read-only and do not update it
    bool ctx_shift         = true;
    bool escape            = true;  // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool multiline_input   = false; // reverse the usage of `\`
    bool simple_io         = false; // improves compatibility with subprocesses and limited consoles
    bool cont_batching     = true;  // insert new sequences for decoding on-the-fly
    bool flash_attn        = true;  // flash attention
    int  mla_attn          = 3;     // MLA 0: standard, 1: MLA with K and V^T cache, 2: MLA with just K cache, 3: the best of both worlds
    int  attn_max_batch    = 256;   // Max batch size to use when computing attention (only applicable if flash_attn = false)
    bool fused_moe_up_gate = true;  // fused up*unary(gate) op for MoE models
    bool fused_up_gate     = true;  // fused up*unary(gate) op
    bool fused_mmad        = true;  // fused mul+multi_add op
    bool grouped_expert_routing = false; // if to use grouped expert routing (BailingMoeV2 arch)
    bool rope_cache        = false; // if to use RoPE cache (for supported models)
    bool graph_reuse       = true;  // if to reuse compute graphs
    bool dsa               = false; // enable GLM DSA sparse attention (off by default; opt-in via --dsa)
    bool fused_idx_topk    = true;  // enable the fused indexer topk op (on by default; -no-fidx or --no-fused-indexer-topk turns it off)
    bool swa_compress      = false;
    bool dsv4_legacy_state = false; // if true, write DeepSeek-V4 state in the legacy full-slice layout (no MAGIC), byte-identical to main
    int  dsa_top_k         = -1;    // DSA top-k override (<0 => use the model's configured indexer_top_k)
    int  min_experts       = -1;
    float thresh_experts   = 0;
    bool no_perf           = false; // disable performance metrics
    std::string ollama_modelfile;                 // llama-server --modelfile: Ollama Modelfile whose PARAMETER/SYSTEM/MESSAGE seed /api/chat + /api/generate
    bool        ollama_strict_names = false;      // llama-server --ollama-strict-model-names: unknown model names answer 404 on /api/*
    std::string ollama_model;                     // --ollama-model NAME: resolve NAME in an Ollama model store (manifest -> GGUF/mmproj/LoRA blobs) and serve it
    std::string ollama_models_dir;                // --ollama-models-dir DIR (default $OLLAMA_MODELS, else ~/.ollama/models)
    bool        ollama_env = false;               // --ollama-env: honour OLLAMA_KV_CACHE_TYPE / FLASH_ATTENTION / CONTEXT_LENGTH / NUM_PARALLEL / KEEP_ALIVE / MAX_LOADED_MODELS
    bool        ollama_store_write = false;       // --ollama-store-write: /api/copy and /api/delete may modify the Ollama store
    bool        ollama_go_template = false;       // --ollama-go-template: translate the store's Go TEMPLATE layer to Jinja when the GGUF has no chat template
    bool show_timings      = true;  // show timing information on CLI
    bool swa_full          = false; // use full-size SWA cache
    bool kv_unified        = false; // enable unified KV cache
    bool kv_admission      = false; // server, --kv-unified only: defer a task whose admission would push the projected peak KV past the pool
    int32_t kv_admission_decode_tokens = 0; // --kv-admission: cap on the decode tokens projected per request (0 = no cap: full n_predict / slot context)
    // theirs' own `ctx_shift = false;` (this block) DROPPED as a duplicate declaration --
    // block 5 already declares this member (=true). The TRUE-vs-FALSE default clash is
    // NAMED, not fixed, per the decision row's explicit instruction.

    bool input_prefix_bos  = false; // prefix BOS to user inputs, preceding input_prefix
    bool verbose_prompt    = false; // print prompt tokens before generation
    bool display_prompt    = true;  // print prompt before generation
    // RESTORED, not disabled (RESEARCH-DONE: git show a4090d1174aed22dde5cacce2a4c27656b987a2f
    // read in full, cross-checked against common/common.cpp.pre-fmt-all which still carries
    // this fork's own un-prefixed llama_kv_cache_dump_view_seqs body -- see that function's
    // definition below for the full account, S69-EXAMPLES-KV-VIEW-DISPOSITION-2026-09-08-01).
    // This fork's own common.h already DECLARED llama_kv_cache_dump_view(_seqs) below
    // (un-prefixed, fork-native spelling) but had no `dump_kv_cache` field to drive it from
    // the CLI -- examples/parallel/parallel.cpp reads params.dump_kv_cache directly, and would
    // not compile without this field. Default false matches upstream's own original default
    // (opt-in debug flag), not a capability being turned off.
    bool dump_kv_cache     = false; // dump the KV cache contents for debugging purposes
    bool no_kv_offload     = false; // disable KV offloading
    bool warmup            = true;  // warmup run
    bool batch_warmup      = false; // batch warmup run
    bool check_tensors     = false; // validate tensor data
    std::string model_checksum;
    bool auto_truncate = false; // server: cut an over-long text prompt (middle dropped) instead of refusing it (--auto-truncate)     // --model-checksum: lower/upper-case hex SHA-256 the main model file must have; empty = no check
    bool repack_tensors    = false; // repack tensors if interleaved variant is available
    bool use_thp           = false; // use transparent huge pages (linux only)
    bool validate_quants   = false; // if true, check for NaNs while loading the model
    bool only_active_exps  = true;  // if true, offload only active experts (relevant only for hybrid CPU/GPU)
    bool merge_qkv         = false; // if true, merge separate Q, K, V tensors into a single, contiguous tensor
    bool merge_up_gate_exps= false; // if true, merge ffn_up_exps and ffn_gate_exps into a single, contiguous tensor
    bool defer_experts     = false; // if true, defer expert mmap residency to speed up model loading (Linux only)
    bool defer_ple         = false; // if true, keep the per-layer token embedding on the file (Linux and Windows only)
    bool defer_ple_auto    = true;  // --lazy-mode auto (upstream default): defer those tables only when they exceed 4 GiB
    bool prefetch_experts  = false; // if true, stream mmap'd MoE expert weights into the page cache (Linux only)
    int  prefetch_experts_threads = 0; // number of expert prefetch workers (<=0 = auto)
    bool k_cache_hadamard  = false; // if true, use Hadamard transform for the K-cache (only makes sense with quantized cache)
    bool v_cache_hadamard  = false; // if true, use Hadamard transform for the V-cache (only makes sense with quantized cache, which requires FA)
    int  kv_fresh_chunk    = 0;     // --kv-fresh-chunk: 1 = current ubatch attends its own unquantized K/V under FA, 2 = control arm, 0 = off
    bool split_mode_graph_scheduling = false; // if true, force split mode graph scheduling
    //bool split_mode_f16    = true;  // if true, intermediate results will be cast to f16 before copying to other GPUs to perform reduce ops
    bool scheduler_async   = false; // if true, in split mode graph the scheduler will use multiple threads to evaluate the graph
    int  fused_delta_net   = 0;     // use fused delta-net if number of tokens in the batch is less than this value
    bool has_mtp           = false; // enable MTP if supported by the model

    std::string cache_type_k = "f16"; // KV cache data type for the K
    std::string cache_type_v = "f16"; // KV cache data type for the V
    std::string indexer_cache_type_k = "f16"; // indexer K-cache data type

    // ── TriAttention KV scoring (unified ik_llama build) ────────────────────
    // RESTORED 2026-08-20 alongside the argv parsing below; the implementation
    // was already compiling into every binary with no way to enable it.
    std::string triattention_stats;  // pre-computed calibration stats (empty = disabled)
    float       tri_budget   = 0.0f; // retention fraction (0.75 = keep 75%), 0 = disabled
    int         tri_window   = 512;  // recent N tokens always kept
    int         tri_interval = 128;  // score/prune every N tokens
    int         tri_sink     = 0;    // first N tokens always kept as attention sinks

    // ── CATS activation sparsity (S69-CATS-STAGE1-2026-09-28) ───────────────
    // arXiv:2404.08763. Stage 1: CPU dense-mask quality path only, no sparse kernel.
    float       cats_sparsity        = 0.0f; // target sparsity in [0,1); 0 = off (default)
    std::string cats_thresholds_file;        // calibration JSON from tools/cats-calibrate (empty = none)
    bool        cats_calibrate       = false; // internal: tools/cats-calibrate sets this directly (not a user CLI flag)
    // ── MoE resident expert pool (S69-KT-EXPERT-POOL-2026-09-28) ────────────
    // Port of the id-remap design shared by two independent upstream sources (attribution
    // required by rule 1.87, cited again at the cparams struct and the call site):
    //   - llama.cpp PR #27861 (csantiago78, MIT, open/draft): per-layer GPU-resident LRU
    //     slots for host-offloaded MoE experts; --moe-expert-cache N; get_rows id->slot
    //     remap; no new ggml ops.
    //   - llama.cpp discussion #28248 (memoriaru, MIT, RFC): same id-remap idea with a
    //     safer prefill/working-set gate (n_expert_used*n_tokens <= n_slots falls back to
    //     the stock path entirely) and the root-caused fix for the map-table race (the
    //     table must be a real buffer-backed tensor, never a view, or a scheduler split can
    //     read it before the update publishes -- discussion #28248 comment, "Root cause").
    // This fork's synchronous variant (see src/llama-moe-pool.h) updates pool residency
    // from the PREVIOUS completed decode step's observed expert ids (captured the same way
    // the recorder below captures them) and finalizes the map table on the host BEFORE the
    // next step's graph is built, so there is no mid-graph race to reproduce in the first
    // place -- it trades the upstream async-upload overlap for that safety margin.
    int         moe_pool_slots      = 0;     // per-layer resident-expert slots (0 = disabled)
    int         moe_pool_inserts    = 2;     // max admissions applied per layer per step
    int         moe_pool_admit      = 3;     // recent-use admission threshold (1 = unconditional, KT/admission-gate idea from the PR thread)
    bool        moe_pool_cpu_only   = false; // restrict the pool to plain CPU/CPU_Mapped buffers, skip device host buffers (e.g. Vulkan_Host) that are already GPU-readable
    std::string moe_stats_path;              // expert-activation recorder output JSON (empty = off)
    std::string moe_stats_mode      = "stat";// stat | stat_approx | per_pass | per_token (KT recorder modes)
    std::string moe_placement_strategy = "uniform"; // uniform | frequency | front-loading | random (KT strategy names)
    std::string moe_placement_stats;         // recorded activation-stats JSON consumed by the frequency strategy
    bool        moe_dynamic_update  = false; // re-place experts once prefill crosses the threshold below
    int         moe_dynamic_threshold = 512; // prefill token threshold for dynamic re-placement

    // FFD block-sparse decode (--ffd): 2-bit per-block K shadow + block selection on decode. BENCH-PENDING.
    bool        ffd       = false;
    int32_t     ffd_block = 128;   // cells per block

    // KIVI simulated low-bit KV (--kivi-bits): 0 = off. Quantises the f16 KV values that leave the residual window.
    int32_t     kivi_bits     = 0;
    int32_t     kivi_group    = 32;
    int32_t     kivi_residual = 128;
    // XAttention prefill block selection (--xattn-threshold): 0 = off. Flash-attention, single-sequence causal prefill.
    float       xattn_threshold = 0.0f;
    int32_t     xattn_stride    = 8;
    int32_t     xattn_block     = 128;
    // KVCrush / AdaptiveR-KV retention for the SnapKV/H2O prompt prune (--kv-diversity-mass): 0 = off
    float       kv_diversity_mass   = 0.0f;
    int32_t     kv_diversity_block  = 16;
    int32_t     kv_diversity_layers = 4;
    // SnapKV/H2O survivor re-base after the prune (--snapkv-rebase): 0 = keep gapped positions
    int32_t     snapkv_rebase = 0;
    float       ffd_delta = 5.0f;  // keep blocks whose estimated max score >= row max - delta (upstream default)

    // ── TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28) ───────────────
    // arXiv:2408.14690. Stage 1: CPU dense-mask quality path only, no sparse kernel;
    // input sparsity on Wq/Wk/Wv/Wgate/Wup/Wdown (Wo excluded this session, see llama.h).
    float       teal_sparsity        = 0.0f; // target sparsity in [0,1); 0 = off (default)
    std::string teal_thresholds_file;        // calibration JSON from tools/cats-calibrate --teal (empty = none)
    bool        teal_calibrate       = false; // internal: calibration tool sets this directly (not a user CLI flag)

    std::string reduce_type = "f16";
    std::string graph_attn_precision = "f16";

    std::string type_k_first = "f16";
    std::string type_k_last  = "f16";
    std::string type_v_first = "f16";
    std::string type_v_last  = "f16";
    int32_t     n_k_first    = -1;
    int32_t     n_k_last     = -1;
    int32_t     n_v_first    = -1;
    int32_t     n_v_last     = -1;

    std::string extra_output_type = "";

    // multimodal models (see examples/mtmd) -- DEDUPED S69-COMMONH-STRUCT-DUP-FIELDS-2026-09-08-01:
    // mmproj/mmproj_use_gpu/no_mmproj/image/image_min_tokens/image_max_tokens were declared TWICE
    // in this struct (once here, once again ~55 lines below under "multimodal models (see
    // tools/mtmd)") -- a genuine duplicate-member compile error, not a conflict-marker miss (git
    // silently concatenated two additive "ported from theirs" blocks with no textual overlap).
    // Kept the fuller/later declaration below (adds mmproj_device, mtmd_batch_max_tokens); this
    // block keeps only the fields that block does NOT also declare.
    std::string mtmd_kq_type = "f32";
    int32_t n_threads_mtmd = -1; // number of threads to use for multimodal processing (-1 = use n_threads_batch, then n_threads)
    bool    send_done      = false;        // send done message as required for OAI compatibility

    // tool call and template -- same dedup: chat_template/use_jinja/enable_chat_template/
    // force_pure_content_parser/reasoning_format/enable_reasoning/prefill_assistant also
    // duplicated further below (server params section); kept there, trimmed here.
    bool use_peg = false;
    std::string system_prompt = "";
    bool parallel_tool_calls = false;
    int reasoning_budget = -1;
    std::string reasoning_budget_message; // message injected before end tag when budget exhausted
    std::map<std::string, std::string> default_template_kwargs;
    std::map<std::string, std::string> force_template_kwargs; // --chat-template-kwargs-force: replace the request's same keys

    thinking_tokens think_tokens;

    bool dry_run              = false;
    int32_t nvfp4_precision   = 0;     // --nvfp4-precision: llama_nvfp4_precision (0 auto, 1 w4a4, 2 int8)
    float k_smooth_alpha      = 0.0f;  // --k-smooth: 0 = off; llama_model_params::k_smooth_alpha

    // --- ported from theirs (block 7 union, additive; ggml_type cache_type_k/v dropped,
    //     SAME-NAME clash with ours' std::string version above -- named in decision row) ---
    bool no_op_offload     = false; // globally disable offload host tensor operations to device
    bool no_extra_bufts    = false; // disable extra buffer types (used for weight repacking)
    bool no_host           = false; // bypass host buffer allowing extra buffers to be used
    bool load_mtp          = false; // load MTP/NextN layers

    bool single_turn       = false; // single turn chat conversation



    common_conversation_mode conversation_mode = COMMON_CONVERSATION_MODE_AUTO;

    // multimodal models (see tools/mtmd)
    struct common_params_model mmproj;
    bool mmproj_use_gpu = true;                 // use GPU for multimodal model
    ggml_backend_dev_t mmproj_device = nullptr; // GPU device to use for multimodal model
    bool no_mmproj = false;                     // explicitly disable multimodal model
    std::vector<std::string> image;             // path to image file(s) ; TODO: change the name to "media"
    int image_min_tokens = -1;
    int image_max_tokens = -1;
    int mtmd_batch_max_tokens = 1024;
    int mmproj_embd_cache_mib = 0;              // --mmproj-embd-cache: MiB of vision-encoder outputs kept for returning images (0 = off)
    int     video_max_frames = 0;               // --video-max-frames: a video request reads at most this many frames (0 = no cap)
    int64_t video_max_tokens = 0;               // --video-max-tokens: a video request's vision tokens must fit this budget; frames are scaled down (0 = no budget)
    // video decode settings (upstream f29551215 / #24318); defaults equal mtmd_helper_video_init_params_default()
    float       video_fps                   = 4.0f; // --video-fps: target frame rate (<= 0 = the clip's native rate)
    int64_t     video_timestamp_interval_ms = 5000; // --video-timestamp-interval: a time-stamp text chunk every N ms (<= 0 = none)
    std::string video_ffmpeg_bin_dir        = "";   // --video-ffmpeg-dir: directory holding ffmpeg + ffprobe (empty = search PATH)


    // finetune
    struct lr_opt lr;
    enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    float val_split = 0.05f; // fraction of the data used for the validation set

    // embedding
    bool embedding         = false; // get only sentence embedding
    bool    embd_sort_by_length = false; // llama-embedding: pack prompts longest first (CTranslate2 rebatch_input); off = caller's order
    bool    restrict_output_vocab = false; // llama-server: multiply the output projection only by the allowed_token_ids of the running requests (CTranslate2 vocabulary map)
    int32_t embd_normalize = 2;     // normalisation for embeddings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
    std::string embd_out   = "";    // empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
    std::string embd_sep   = "\n";  // separator of embeddings
    std::string cls_sep    = "\t";  // separator of classification sequences

    // server params
    int32_t port                = 9931;          // server listens on this network port
    bool    reuse_port          = false;         // allow multiple sockets to bind to the same port
    int32_t timeout_read        = 3600;          // http read timeout in seconds
    int32_t timeout_write       = timeout_read;  // http write timeout in seconds
    int32_t sse_ping_interval   = 30;            // SSE ping interval in seconds
    int32_t max_queued_requests = 0;             // admission: completion requests in flight at once (0 = no limit)
    int64_t max_queued_tokens   = 0;             // admission: prompt tokens of requests still in prefill (0 = no limit)
    int64_t rate_limit_calls     = 0;            // --rate-limit CALLS/PERIOD: POST requests per client per window (0 = no limit)
    int64_t rate_limit_period_ms = 0;
    std::string rate_limit_key   = "addr";       // --rate-limit-key: count per remote address ("addr") or per API key ("key")
    int32_t max_request_size_mib = 100;          // --max-request-size: HTTP request body cap in MiB, 413 above it (0 = no cap)
    int32_t n_threads_http      = -1;    // number of threads to process HTTP requests (TODO: support threadpool)
    int32_t n_cache_reuse       = 0;     // min chunk size to reuse from the cache via KV shifting
    bool    cache_prompt        = true;  // whether to enable prompt caching
    bool    cache_idle_slots    = true;  // save and clear idle slots upon starting a new task
    bool    cache_publish_on_prefill = true;  // proactively publish a slot's KV to the prompt cache
                                               // the instant its own prefill finishes, so a
                                               // concurrently-active sibling slot can reuse the
                                               // shared prefix without waiting for this slot to go
                                               // idle (SGLang RadixAttention gap port). Always a
                                               // no-op when n_parallel == 1: no sibling can ever
                                               // consume the publish, so it is pure serialization
                                               // cost with zero possible benefit.
    int32_t n_ctx_checkpoints   = 32;    // max number of context checkpoints per slot
    int32_t kv_unified_per_slot = 0;     // max context per parallel slot; 0 = unset
    int32_t checkpoint_min_step = 8192;  // minimum spacing between context checkpoints
    int32_t cache_ram_mib       = 8192;  // -1 = no limit, 0 - disable, 1 = 1 MiB, etc.
    // RESEARCH-DONE: engines/vllm/vllm/v1/kv_offload/cpu/policies/arc.py, engines/sglang/python/sglang/srt/mem_cache/utils.py:48-84 (read)
    bool    cache_ram_slru      = false; // scan-resistant (segmented LRU) eviction of the host prompt cache
    // Embedding batches: post the longest input first (first-fit-decreasing packing of micro-batches); results keep request order.
    bool    embd_length_order   = true;
    // Grammar jump-forward (row S69-MLC-LLM-PICK-GRAMMAR-JUMP-FORWARD-2026-10-15-01): when a user GBNF grammar leaves no choice about
    // the next text, up to this many tokens spelling it are appended and decoded in ONE batch instead of one decode per token.
    // 0 = off. A request can override it with "grammar_jump_forward".
    int32_t grammar_jump_forward = 0;
    // Strict JSON schema (row S69-SGLANG-PICK-STRICT-JSON-SCHEMA-CHECK-2026-10-18-01): a json_schema whose constraints the grammar
    // converter would drop or weaken (common/json-schema-strict.h) is refused with 400 instead of accepted. A request can override
    // it with "json_schema_strict". false = accept and log the dropped constraints.
    bool    json_schema_strict  = false;
    // Startup-warmed, eviction-pinned prefixes (row S69-rtp-llm-PICK-STARTUP-WARMED-PINNED-PREFIX-CACHE-2026-09-29-01):
    // a JSON list [{"id", "messages" | "prompt"}] is run through the normal request path once after the model loads and the
    // resulting prompt states are pinned in the host prompt cache. Pinned bytes may hold at most warm_prefix_share of --cache-ram.
    std::string warm_prefix_file;
    float       warm_prefix_share = 0.5f;
    // NVMe tier under the host prompt cache (common/disk-tier.h; row S69-TIERED-KV-DISK-TIER-SLICE1-2026-10-05-01).
    // A prompt state the RAM cache must evict is written to this directory instead of dropped, and read back (checksum
    // verified) when a later request's prefix matches it. Empty dir = off, and then behaviour is byte-identical to before.
    std::string cache_disk_dir = "";
    int32_t cache_disk_mib     = 8192; // hard cap on the bytes in the directory
    // free-space floor of the tier's volume (common_disk_tier::set_free_space_policy): below it the tier writes nothing.
    // -1 = the built-in floor (common_disk_tier::DEFAULT_MIN_FREE_BYTES, 10 GiB), 0 = floor off, N > 0 = keep N MiB free.
    int32_t cache_disk_min_free_mib = -1;
    // server beam search (request field use_beam_search): which ranking/stopping rules the loop follows.
    //   "vllm" (default) = vLLM's loop, byte-identical to before; "ct2" = CTranslate2's (SOURCE-READ src/decoding.cc:411-713).
    // A request's `beam_mode` overrides it.
    std::string beam_mode = "vllm";
    // MEASURED (temp/_c2_1005_rb_*.txt, tests/bench-tier-restore.cpp, CPU, this host): restoring a state from the tier beats
    // recomputing it from about 3 tokens (OLMo-2-1B) to 20 tokens (granite-4.0-h-350m), over 4 models; 32 is above all of them.
    // The earlier default, 256 (SGLang HiCache's prefetch threshold), was not a number measured here.
    int32_t cache_disk_min_tokens = 32;  // never spill a prompt shorter than this: re-prefilling it is cheaper than the I/O
    // restore-versus-recompute rule (server_prompt_cache::restore_pays): a spill/restore must be at least this many times quicker
    // than recomputing, by the speeds measured on the running server. 0 = rule off.
    float   cache_disk_min_speedup = 1.0f;
    int32_t cache_disk_write_mib_per_hour = 0; // SSD wear budget, 0 = unlimited (a token bucket holding one hour)
    int32_t cache_disk_io_chunk_mib = 0; // >0: overlap the tier's file I/O with its checksum in chunks of this size (default 0 = serial)

    // On-device token-level radix prefix cache (row S69-ADD-TGI-ALTERNATIVES-...-RADIX-2026-09-27-01,
    // parent design S69-ENGINE-WAVE2-SGLANG-RADIXATTENTION-2026-09-26-01). Distinct from
    // cache_publish_on_prefill/server_prompt_cache above: that mechanism serializes a slot's WHOLE
    // state to host RAM and restores it for an EXACT-match sibling; this one shares KV at TOKEN
    // granularity, on-device, zero-copy (llama_memory_seq_cp within one --kv-unified stream only
    // tags existing cells, src/llama-kv-cache.cpp:527-556 -- no data movement), so two DIFFERENT
    // prompts that merely share a prefix (e.g. a common system prompt) both benefit, not just an
    // identical resend.
    // DISABLE-REVERSIBLE: default OFF is this NEW feature's own opt-in switch (--radix-cache), not
    // a working feature being turned off -- mirrors n_cfg_guidance_slots=0 below (same pattern,
    // row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01). Re-enable condition: pass
    // --radix-cache. Reserves extra seq ids (widens n_seq_max, same mechanism as
    // n_cfg_guidance_slots) whether or not a request ever uses them, which is why it defaults off.
    bool    radix_cache          = false; // opt-in; requires --kv-unified (checked at server init)
    int32_t n_radix_cache_slots  = 32;    // reserved seq ids for the tree's own cached nodes
    // Which idle leaf the radix tree gives up first when its reserved seq ids run out: lru (default, the historical behaviour), lfu, fifo,
    // mru, filo, slru, priority (SGLang mem_cache/evict_policy.py). Only read when radix_cache is on.
    std::string radix_evict_policy = "lru";
    // Which deferred request starts next when a slot frees up: fcfs (lowest `priority`, then arrival; the historical behaviour) or lpm (within a
    // priority, longest already-cached prefix first, and a request sharing a long prefix with an earlier waiting one goes last; SGLang
    // managers/schedule_policy.py).
    std::string sched_policy = "fcfs";
    int32_t     sched_prefix_threshold = 32; // lpm: tokens of shared prefix that count as "worth waiting for"; -1 turns the in-batch rule off
    int32_t     radix_evict_slru_hits = 2; // slru: nodes inserted through at least this many times are protected

    // ---- serving-feature wiring (docs/agent-reports/wire-serving-features.md). Every field defaults OFF; OFF is bit-identical to
    //      a build without the feature. ----
    // LoRA adapter pool: a request names an adapter ("lora_adapter":"<name>"), the server loads <lora_pool_dir>/<name>.gguf on demand,
    // keeps at most lora_pool_max adapters (and lora_pool_max_mib MiB of adapter files) resident, evicts the least recently used one that no
    // request is running with, and answers 503 when every resident adapter is pinned by a running request.
    std::string lora_pool_dir;                 // --lora-pool-dir; empty = pool off
    int32_t     lora_pool_max     = 4;         // --lora-pool-max; 0 = no count limit
    int32_t     lora_pool_max_mib = 0;         // --lora-pool-max-mib; 0 = no byte limit
    bool        lora_hetero_batch = false;     // --lora-hetero-batch: slots with different adapters share one decode batch (per-sequence LoRA)
    // koboldcpp-port (behaviour spec: docs/agent-reports/koboldcpp-spec.md, no code copied)
    bool        token_ban_substr  = false;     // --token-ban-substr: honour the request field "banned_tokens" (vocab-substring ban)
    bool        prompt_reuse_plan = false;     // --prompt-reuse-plan: size the slot-prompt reuse with prompt_reuse_plan() (common/prompt-reuse.h)
    int32_t     prompt_reuse_min  = 0;         // --prompt-reuse-min: reuse below this many tokens is discarded
    int32_t     prompt_reuse_keep_input = 0;   // --prompt-reuse-keep-input: always leave at least this many prompt tokens to be evaluated
    // ASR decode quality on /v1/audio/transcriptions (examples/audio/asr-fallback.h, asr-compression.h; whisper.cpp / faster-whisper semantics)
    bool        asr_quality       = false;     // --asr-quality: gate each transcription and retry at a higher temperature when it fails
    float       asr_temperature_inc       = 0.2f;   // --asr-temperature-inc
    float       asr_compression_ratio_thold = 2.4f; // --asr-compression-ratio-thold
    float       asr_logprob_thold         = -1.0f;  // --asr-logprob-thold
    float       asr_entropy_thold         = 2.4f;   // --asr-entropy-thold (0 = gate off)
    std::string asr_native_prompt         = "off";  // --asr-native-prompt on|off|vibevoice: [INST]<audio>[/INST]lang:xx[TRANSCRIBE] when the vocab has [TRANSCRIBE]
    // --asr-vad: native Silero VAD gate on /v1/audio/transcriptions (examples/audio/asr-vad-gate.h); empty = off (default)
    std::string asr_vad_model             = "";     // --asr-vad FNAME (whisper.cpp ggml-format Silero file)
    float       asr_vad_threshold         = 0.5f;   // --asr-vad-threshold (asr_vad_post_params::threshold default)
    // --asr-vad-trim: with --asr-vad, give the model the speech spans only and map the reply's times back to the upload
    // (examples/audio/asr-vad-trim.h); off (default) = the whole upload goes to the model as before
    bool        asr_vad_trim              = false;  // --asr-vad-trim
    int32_t     asr_vad_trim_pad_ms       = 100;    // --asr-vad-trim-pad-ms (silence between kept pieces; 0 = contiguous)
    bool    cache_ram_explicit  = false; // set by --cache-ram, decision models disable the cache otherwise
    // RESEARCH-DONE: arxiv.org/abs/2505.24782 (ModernBERT full-document pass beats sliding windows), arxiv.org/pdf/2403.00784 (MaxP/SumP passage aggregation)
    // --decision-budget auto|model|N (/v1/systemone): auto = never cut question/options/state, size the context to the model and
    // window a state that does not fit; model = the trained head budget (reference parity); N = explicit head budget in tokens
    std::string decision_budget         = "auto";
    // --decision-window-agg max|mean|lse|first: how the per-option logits of the windows of a long state are combined
    std::string decision_window_agg     = "max";
    // --decision-window auto|N: the longest prompt of one pass (auto = what the context takes; N = windows of N tokens even if more would fit)
    std::string decision_window         = "auto";
    bool    n_ubatch_explicit   = false; // set by -ub, auto budget sizes n_ubatch to the context otherwise

    // llama-connect params
    bool        server_connect      = false;
    std::string server_connect_code = "";

    std::string public_path   = "";                                                                         // NOLINT
    std::string api_prefix    = "";                                                                         // NOLINT
    std::string chat_template = "";                                                                         // NOLINT
    std::vector<std::string> hostnames = {"127.0.0.1"};
    bool use_jinja = true;                                                                                  // NOLINT

    // server CORS params
    std::string cors_origins = "*";
    std::string cors_methods = "GET, POST, DELETE, OPTIONS";
    std::string cors_headers = "*";
    bool cors_credentials = true;
    bool cors_origins_explicit = false; // for --agent option
    std::vector<std::string> allowed_hosts; // extra Host header names accepted on a loopback server with no api key ("*" = any); see server-hostguard.h

    bool enable_chat_template = true;
    bool force_pure_content_parser = false;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    int enable_reasoning = -1; // -1 = auto, 0 = disable, 1 = enable
    bool prefill_assistant = true; // if true, any trailing assistant message will be prefilled into the response
    bool tool_arg_coercion = true; // repair parsed tool-call argument types to the request's tool schemas (non-streaming; a request can override)
    std::string tool_call_format;  // operator override of the XML tool-call format: JSON of common_chat_xml_tool_format (empty = none)
    int sleep_idle_seconds = -1;   // if >0, server will sleep after this many seconds of idle time
    int forward_watchdog_soft_s = 0; // >0: log when no decode step finished for this long while a batch is in flight
    int forward_watchdog_hard_s = 0; // >0: end the process after this long (plus a grace) with no decode progress
    int kv_audit_interval = 0;     // >0: every N decode steps run the KV canary / leak / non-finite-logit audit (debug)

    std::vector<std::string> api_keys;
    std::vector<std::string> api_keys_admin;        // --api-key-admin / --api-key-admin-file: also open the management routes
    std::vector<std::string> api_key_files;         // --api-key-file paths, re-read when they change
    std::vector<std::string> api_key_admin_files;   // --api-key-admin-file paths, re-read when they change

    std::string ssl_file_key  = "";                                                                         // NOLINT
    std::string ssl_file_cert = "";                                                                         // NOLINT

    bool preserve_reasoning_specified = false;


    // "advanced" endpoints are disabled by default for better security
    common_webui webui = COMMON_WEBUI_AUTO;
    bool webui_mcp_proxy  = false;

    // --- ported from theirs (block 8, additive; default_template_kwargs dropped, already
    //     present via block 7 on ours' side; ui/ui_mcp_proxy dropped as a probable but
    //     UNCONFIRMED duplicate of ours' webui/webui_mcp_proxy above -- ESCALATED, not
    //     fixed, per the DEGRADED-PROXY-CANDIDATE instruction) ---
    std::string server_base; // if set, connect to this server instead of starting a new one
    std::string ui_config_json;
    bool endpoint_slots   = true;
    bool endpoint_props   = false; // only control POST requests, not GET
    bool endpoint_metrics = false;
    bool endpoint_memory  = false; // GET /memory (ik_llama.cpp 55813a5d7)

    // enable built-in tools
    std::vector<std::string> server_tools;
    std::string server_tools_runtime;
    size_t      server_tools_max_output = 0; // bytes one tool call may return to the model; 0 = a share of the context size

    // MCP server configs (Cursor-compatible JSON)
    std::string mcp_servers_config;   // path to JSON file with MCP server definitions
    std::string mcp_servers_json;     // inline JSON with MCP server definitions

    // router server configs
    std::string models_dir    = "";     // directory containing models for the router server
    std::string models_preset = "";     // directory containing model presets for the router server
    int models_max = 4;                 // maximum number of models to load simultaneously
    float models_vram_admit = 0.0f;     // router: admit a load only if predicted VRAM <= this x free, evicting idle LRU (0 = off)
    int models_memory_margin = 0;       // router: MiB of device memory to keep free when admitting a load (0 = off), ggml-org #21231
    bool models_autoload = true;        // automatically load models when requested via the router server
    bool models_register = true;        // router: allow POST /models (download + register) and DELETE /models (remove) at runtime
    std::vector<std::string> models_register_allow; // router: regexes (whole-name, case-insensitive) a POST /models repo must match; empty = any
    int64_t models_keep_alive_ms = -1;  // idle time before the router unloads a model (-1 = never), per-request keep_alive overrides
    int64_t models_load_cooldown_ms     = 10 * 1000;   // after a model fails to load, refuse it this long (doubles per consecutive failure); 0 = not in use
    int64_t models_load_cooldown_max_ms = 300 * 1000;  // ceiling of the doubling
    std::string models_preset_hf = "";  // show a warning about remote presets on router loaded (if not empty)
    bool models_reload_allow_empty = false; // router: let a reload that yields zero presets unload every running model (default: refuse)
    std::string models_snapshot = "";   // router: file the loaded-model set is written to on load/unload and restored from at start (empty = off)

    std::string reasoning_effort_budget = ""; // OpenAI reasoning_effort -> thinking budget table ("default" or "low=0.25,high=0.75"), empty = effort only reaches the chat template
    std::string reasoning_answer_reserve = "-1"; // --reasoning-answer-reserve N|P%: thinking budget = generation limit - reserve when a request sets none (server-answer-reserve.h), "-1" = off

    // reasoning-loop recovery (tools/server/server-loop-recovery.h): what a loop found inside the reasoning does when the
    // request does not say: "off" (default), "stop" or "recover"; and the effort sentences the recover retry swaps
    // (empty = the Qwen3.8 chat template's xhigh / low sentences)
    std::string reasoning_loop_recovery    = "off";
    std::string reasoning_effort_high_text = "";
    std::string reasoning_effort_low_text  = "";

    bool sampling_family_defaults = false;    // --sampling-family-defaults: vendor-recommended sampling per model family when the GGUF has no general.sampling.* key
    int32_t     draft_max_by_slots_hysteresis = 0; // --draft-max-by-slots-hysteresis: slots of margin before the budget may rise again (0 = off)
    std::string draft_max_by_slots;        // --draft-max-by-slots: "1:5,3:2,9:0" draft length by the number of generating slots (empty = off)
    std::string sampler_overrides_file;      // --sampler-overrides: JSON layer of operator request-field overrides (default / force / additive)
    std::string sampler_overrides_model_file; // --sampler-overrides-model: per-model layer on top of it
    bool server_disable_management_routes = false; // refuse the routes that act on the server itself (control vectors, LoRA, slot save/restore/erase, SQL) with 403

    bool log_json = false;

    std::string slot_save_path;
    std::string sql_save_file;
    std::string sqlite_zstd_ext_file;

    float slot_prompt_similarity = 0.1f;
    int32_t slot_evict_lookahead = 0; // --slot-evict-lookahead: waiting requests checked before an idle slot is evicted (0 = plain LRU)

    bool do_checkpoint = false;               // do checkpoint for recurrent models only
    bool        ctx_checkpoint_cram_soft = false; // --ctx-ckpt-cram-soft: cold checkpoints go to the compressed block store (llama_cram_soft_store) instead of staying raw in RAM
    bool        cram_probe = false;           // --cram-probe: log what the read-only sysfs probe finds at model load (diagnostic only)
    // --vram-spill (default off): where large device allocations go when VRAM is short. See ggml/include/ggml-vram-spill.h.
    std::string vram_spill_mode      = "off"; // off | managed | hostmap | auto
    int64_t     vram_spill_threshold_mib = 256; // allocations smaller than this never spill
    int64_t     vram_spill_reserve_mib   = 1024; // auto: VRAM kept free
    std::string ctx_checkpoint_spill_dir = ""; // disk dir for checkpoint spill (empty = disabled, behavior unchanged)
    bool        ctx_checkpoint_dominance = false; // --ctx-ckpt-dominance: at slot release free checkpoints dominated by a deeper one (common/ckpt-dominance.h)
    int32_t ctx_checkpoint_ram_live = 2;    // max checkpoints with resident data when spill is on
    int32_t ctx_checkpoints_tolerance = 5;    // the number of tokens before the full prompt to create the checkpoint
    bool    prefill_fair_share = false;       // split the prompt tokens of one batch across the slots that are still prefilling (max-min fair) instead of first slot takes all
    int32_t prefill_step_budget = 0;                  // SARATHI-style token budget: when decode tokens ride in the step, all slots together add at most this many prompt tokens (0 = off). Pure-prefill steps keep the whole batch
    int32_t prefill_tbt_target_ms = 0;                // adaptive version: size that budget each step from the measured prefill us/token and decode-step time so a mixed step stays near this many ms (0 = off; --prefill-step-budget is the cold-start value)
    bool    prefill_ubatch_chunks = false;    // add at most n_ubatch prompt tokens per slot per scheduler step, so a cancel/disconnect lands between micro-batches instead of after a whole n_batch decode (fastllm C9)
    std::string control_vector_root = "";     // POST /control-vectors/load only accepts files under this directory (empty = any readable path)
    int32_t stream_interval = 1;            // streaming: send a chunk only once N generated tokens are unsent (1 = every token, as before); the last chunk is always sent
    int32_t max_queue_wait_ms = 0;        // a request still waiting for a slot after this many ms is answered 503 and dropped (0 = wait forever; a request can set its own)
    bool    evict_low_priority = false;     // a higher-priority arrival on a full admission queue evicts the lowest-priority queued request (503 "evicted")
    int32_t kv_reserve_ratio = 0;           // --kv-unified: percent of the KV pool held back from NEW requests so running ones can grow (0 = off)
    bool    kv_preempt_requeue = false;     // --kv-unified: a NON-streaming request preempted on KV exhaustion is requeued and recomputed instead of answered 503
    int32_t kv_preempt_requeue_max = 3;     // most times one request is requeued before the next preemption answers 503
    int32_t kv_preempt_victim = 0;          // --kv-unified: which in-flight request a KV-exhaustion preemption evicts: 0 largest (default), 1 least-progress, 2 newest
    int32_t kv_preempt_near_finished = 0;   // --kv-preempt-near-finished N: a request with <= N tokens left to generate is not chosen as the preemption victim while another exists (0 = off)
    std::string request_log_path;           // --request-log FILE: one JSONL line per finished request (empty = off)
    double      request_log_rate = 1.0;     // --request-log-rate: fraction of requests logged
    std::string request_log_model_rate;     // --request-log-model-rate NAME=R,...: per-model override
    int32_t shed_max_latency_ms = 0;       // predictive shedding: drop the oldest queued request when wait + predicted service time reaches this (0 = off)
    std::string fallback_model = "";        // router mode: comma-separated fallback chain of this model entry
    int32_t failover_max_attempts = 0;      // router mode: most models tried per request (0 = whole chain)
    int32_t request_deadline_ms = 0;        // end-to-end budget of a completion request, counted from its arrival: queue wait + prompt + generation (0 = none; a request can set its own)
    common_checkpoint_eviction ctx_checkpoint_eviction = COMMON_CHECKPOINT_EVICTION_VARIANCE;
    // cache_ram_mib DEDUPED (S69-COMMONH-STRUCT-DUP-FIELDS-2026-09-08-01): declared again above
    // in the server-params section (int32_t cache_ram_mib = 8192) -- same name+type+default,
    // second declaration removed here.
    int32_t cache_ram_n_min = 0;     // min number of tokens required to save in the ram
    float cache_ram_similarity = 0.5f; // similarity of tokens to cached tokens
    std::string media_path; // path to directory for loading media files

    // batched-bench params
    bool is_pp_shared   = false;
    bool is_tg_separate = false;

    std::vector<int32_t> n_pp;
    std::vector<int32_t> n_tg;
    std::vector<int32_t> n_pl;

    // retrieval params
    std::vector<std::string> context_files; // context files to embed

    int32_t chunk_size = 64; // chunk size for context embedding

    std::string chunk_separator = "\n"; // chunk separator for context embedding

    // passkey params
    int32_t n_junk = 250; // number of times to repeat the junk text
    int32_t i_pos  = -1;  // position of the passkey in the junk text

    // imatrix params
    std::string out_file = "imatrix.dat"; // save the resulting imatrix to this file
    std::string out_file_draft = "";      // optional paired draft imatrix output file
    std::string output_tensor_name = "output.weight"; // name of the output tensor

    int32_t n_out_freq  = 10; // output the imatrix every n_out_freq iterations
    int32_t n_save_freq =  0; // save the imatrix every n_save_freq iterations
    int32_t i_chunk     =  0; // start processing from this chunk
    int8_t  imat_dat    =  0; // whether the legacy imatrix.dat format should be output (gguf <= 0 < dat)

    bool process_output  = false; // collect data for the output tensor
    bool compute_ppl     = true;  // whether to compute perplexity
    bool show_statistics = false; // show imatrix statistics per tensor
    bool activation_statistics = false; // generate data to calculate activation based statistics
    bool parse_special   = false; // whether to parse special tokens during imatrix tokenization
    // PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01:
    // examples/imatrix/imatrix.cpp:251,267 (fork-only -lsim/--layer-similarity). Field lives on
    // common_params (not a local main() bool like the examples file) because the collector reads
    // it via m_params after common_params_parse, matching how output_tensor_name above is read.
    bool collect_lsim    = false; // collect layer/attn/ffn cosine-similarity importance data

    // --hessian-block N: also accumulate the N x N diagonal blocks of X^T X per matrix into a sidecar GGUF
    // (common/imatrix-hessian.h; row S69-F1-PICK-IMATRIX-BLOCK-HESSIAN-OUTPUT-2026-10-06-01). 0 = off.
    int32_t     imat_hessian_block   = 0;
    std::string imat_hessian_out     = "";   // sidecar path; empty = <imatrix out without .gguf>.hessian.gguf
    int32_t     imat_hessian_max_mib = 4096; // refuse to start a matrix that would take the total past this

    // cvector-generator params
    int n_pca_batch = 100;
    int n_pca_iterations = 1000;
    dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
    std::string cvector_outfile       = "control_vector.gguf";
    std::string cvector_positive_file = "tools/cvector-generator/positive.txt";
    std::string cvector_negative_file = "tools/cvector-generator/negative.txt";

    bool spm_infill = false; // suffix/prefix/middle pattern for infill

    std::string lora_outfile = "ggml-lora-merged-f16.gguf";

    bool sweep_bench_output_jsonl = false;
    bool minilog = false;

    // --- ported from theirs (block 11 union, additive) ---
    bool batched_bench_output_jsonl = false; // DEGRADED-PROXY-CANDIDATE vs ours'
                                              // sweep_bench_output_jsonl above --
                                              // NOT traced to the bench tool source,
                                              // both kept rather than guessed (grant).
    bool tokenize_ids        = false;
    bool tokenize_stdin      = false;
    bool tokenize_no_bos     = false;
    bool tokenize_show_count = false;
    llama_progress_callback load_progress_callback = NULL;
    void *                  load_progress_callback_user_data = NULL;
    std::string tts_lang = "";
    std::string tts_speaker_file = "";
    bool is_gen_docs = false; // whether we are running inside llama-gen-docs
};

// --cache-disk-io-chunk: the chunk size in bytes the disk tier is told to use (0 = serial I/O). One definition so the server and the tests agree.
inline size_t common_cache_disk_io_chunk_bytes(const common_params & params) {
    return params.cache_disk_io_chunk_mib > 0 ? (size_t) params.cache_disk_io_chunk_mib * 1024ull * 1024ull : 0;
}


std::pair<int, char**> parse_command_line(const std::string& commandLine);
void free_command_line(int argc, char** argv);

void gpt_params_handle_hf_token(common_params & params);
void gpt_params_parse_from_env(common_params & params);
void gpt_params_handle_model_default(common_params & params);

// RESTORED 2026-09-13. The DEFINITION never went anywhere -- common_init() is at
// common/common.cpp:875 and has been all along -- but its declaration was lost from this header,
// so every caller outside common.cpp saw "'common_init' was not declared in this scope; did you
// mean 'common_log_init'?". That was 5 of the 242 remaining full-tree error lines on its own
// (tools/batched-bench, tools/perplexity, tools/fit-params, tests/test-recurrent-state-rollback,
// and more behind them).
//
// RECOVERED VERBATIM, comment included, from common/common.h.pre-d5:1298-1300 -- this repository's
// own pre-merge copy of this header. How it was lost is visible one line further down in that
// file: a `>>>>>>> unsloth/master` conflict marker sits immediately after the declaration, so the
// block was dropped while resolving that conflict rather than deliberately removed.
//
// call once at the start of a program if it uses libcommon
// initializes the logging system and prints info about the build
void common_init();

void common_params_print_info(const common_params & params, bool print_devices = true);
std::string common_params_get_system_info(const common_params & params);

bool parse_cpu_range(const std::string & range, bool(&boolmask)[GGML_MAX_N_THREADS]);
bool parse_cpu_mask(const std::string & mask, bool(&boolmask)[GGML_MAX_N_THREADS]);
void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model = nullptr);
bool set_process_priority(enum ggml_sched_priority prio);


// REMOVED 2026-09-13: an OLDER common_remote_params and its common_remote_get_content declaration
// stood here. download.h:30 defines the same struct, and any translation unit including both got
//     download.h:30: error: redefinition of 'struct common_remote_params'
//
// download.h's is the one that survives, and the two are NOT interchangeable -- this one held
// `std::vector<std::string> headers`, while download.h:31 holds `common_header_list headers`, which
// download.h:12 defines as `std::vector<common_header>`: structured headers rather than raw
// strings. Keeping the older definition would have silently given some translation units the
// weaker type depending on include order.
//
// download.h is the owner: it also carries common_download_progress, the callback interface and
// common_remote_get_content. Anything needing these includes "download.h" -- as arg.h, arg.cpp,
// download.cpp and preset.cpp already do.

//
// String utils
//
std::string string_join(const std::vector<std::string>& values, const std::string& separator);
std::string string_strip(const std::string & str);
std::string string_get_sortable_timestamp();
std::string string_lower(const std::string & str);
std::string string_repeat(const std::string & str, size_t n);

// REMOVED 2026-09-14: `static bool string_starts_with(const std::string&, const std::string&)`
// stood here WITH a body while `inline bool string_starts_with(std::string_view, std::string_view)`
// and its char-prefix sibling are defined further down this file. A `char *` argument converts
// equally well to std::string and to std::string_view, so every such call was ambiguous:
//     server-models.cpp:1070: error: call of overloaded
//                             'string_starts_with(char*&, const char [27])' is ambiguous
// The string_view pair is upstream's form, is what callers were compiled against, and also covers
// the char-prefix case this one could not. Removing this leaves exactly one viable candidate.
// Same shape, same file, same cause as the string_ends_with and string_split removals below.

std::vector<std::string> string_split(const std::string& str, const std::string& delimiter);
// REMOVED 2026-09-13: `std::vector<std::string> string_split(const std::string&, char delim);`
// stood here with NO DEFINITION anywhere in the tree, and it is a BETTER MATCH for a char argument
// than the template below, so every `string_split(s, ';')` bound to it and failed at link:
//     undefined reference to `string_split(std::__cxx11::basic_string<...> const&, char)'
// 24 references, all from common/sampling.cpp:1312 via every target that links that object.
// The definition that DOES exist is the inline explicit specialization string_split<std::string>
// further down this file, and the two call sites now name it explicitly (common/sampling.cpp:1312,
// examples/quantize/quantize.cpp:388). Deleting the declaration rather than writing a body avoids
// putting a second implementation of one split beside the template's.
// Same shape, same file, same day as the string_ends_with removal below.

void string_replace_all(std::string & s, const std::string & search, const std::string & replace);
// While we wait for C++20's std::string::ends_with...
// REMOVED 2026-09-13: `bool string_ends_with(const std::string_view&, const std::string_view&);`
// stood here and had NO DEFINITION anywhere in the tree -- a scan of common/ finds this function
// only USED (chat.cpp:997, :1524, :1553; chat-diff-analyzer.cpp:556, :563) and defined once, as the
// by-value inline at common.h:1028. Two declarations with different parameter forms are two
// OVERLOADS, so every call became ambiguous:
//     common.h:1034: error: call of overloaded 'string_ends_with(std::string&, std::string_view&)'
//                    is ambiguous
// Deleting the body-less declaration leaves exactly one candidate. The by-value inline is upstream's
// form and is the one every caller was already compiled against.
// REMOVED 2026-09-14: `size_t string_find_partial_stop(const std::string_view&, const std::string_view&);`
// stood here with NO DEFINITION anywhere in the tree -- checked with a positive control
// (string_repeat, declared in this same block, IS defined at common.cpp:1046, so a .cpp definition
// would have been found). The real one is the by-value inline further down this file. Two
// declarations with different parameter forms are two OVERLOADS, so every call became ambiguous:
//     server-context.cpp:536: error: call of overloaded
//                             'string_find_partial_stop(const std::string&, const std::string&)'
//                             is ambiguous
// Deleting the body-less declaration leaves exactly one candidate. Third instance of this exact
// defect in this header; see the string_ends_with and string_split removals above.

std::string regex_escape(const std::string& s);

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LLAMA_COMMON_ATTRIBUTE_FORMAT(...)
#endif

LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
std::string string_format(const char * fmt, ...);

std::string string_strip(const std::string & str);
std::string string_get_sortable_timestamp();
std::string string_lcs(std::string_view a, std::string_view b);

std::string string_join(const std::vector<std::string> & values, const std::string & separator);
std::vector<std::string> string_split(const std::string & str, const std::string & delimiter);
std::string string_repeat(const std::string & str, size_t n);

void string_replace_all(std::string & s, const std::string & search, const std::string & replace);

std::string regex_escape(const std::string & s);

template<class T>
static std::vector<T> string_split(const std::string & str, char delim) {
    static_assert(!std::is_same<T, std::string>::value, "Please use the specialized version for std::string");
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream token_stream(token);
        if (!(token_stream >> value)) {
            throw std::invalid_argument("invalid value: \"" + token + "\"");
        }
        values.push_back(value);
    }
    return values;
}

// Fork helper the upstream merge 279b4bfaf dropped with the old parser (common.cpp @ 35f0cac75:885);
// "a,b,c,d" -> {(a,b),(c,d)}. Used by restored options in arg-fork-restored.inc (-gfm and two more).
template<class T1, class T2>
static std::vector<std::pair<T1, T2>> string_split_pairs(const std::string & str, char delim) {
    std::vector<std::pair<T1, T2>> values;
    std::istringstream str_stream(str);
    std::string token;
    T1 first_value{};
    int i = 0;
    while (std::getline(str_stream, token, delim)) {
        std::istringstream token_stream(token);
        if (i % 2 == 0) {
            token_stream >> first_value;
        } else {
            T2 value{};
            token_stream >> value;
            values.emplace_back(first_value, value);
        }
        i++;
    }
    return values;
}

// REMOVED 2026-09-13: a SECOND explicit specialization of string_split<std::string> stood here,
// byte-different from the one below only in parameter names -- `(input, separator)` against
// `(str, delim)` -- and identical in behaviour. The merge landed both, which is a redefinition:
//     common.h:1044: error: redefinition of 'std::vector<T> string_split(const std::string&, char)
//                    [with T = std::string]'
// The one that survives is the one below, because it is declared `inline`. This one was not, and a
// non-inline function definition in a header is an ODR violation across translation units the
// moment more than one includes it -- it happened to be masked here only because the redefinition
// error fired first.

// `template<>` is required: this is an explicit specialization of the string_split template above,
// and without it the compiler reads it as a redefinition of a member and says so exactly
// ("specializing member '::string_split<std::string>' requires 'template<>' syntax").
template<>
inline std::vector<std::string> string_split<std::string>(const std::string & str, char delim)
{
    std::vector<std::string> parts;
    size_t begin_pos = 0;
    size_t delim_pos = str.find(delim);
    while (delim_pos != std::string::npos) {
        std::string part = str.substr(begin_pos, delim_pos - begin_pos);
        parts.emplace_back(part);
        begin_pos = delim_pos + 1;
        delim_pos = str.find(delim, begin_pos);
    }
    parts.emplace_back(str.substr(begin_pos));
    return parts;
}

// remove when moving to c++20
inline bool string_starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

// remove when moving to c++20
inline bool string_starts_with(std::string_view str, char prefix) {
    return !str.empty() && str.front() == prefix;
}

// remove when moving to c++20
inline bool string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool string_remove_suffix(std::string & str, std::string_view suffix) {
    if (string_ends_with(str, suffix)) {
        str.resize(str.size() - suffix.size());
        return true;
    }
    return false;
}

inline size_t string_find_partial_stop(std::string_view str, std::string_view stop) {
    if (!str.empty() && !stop.empty()) {
        const size_t max_len = std::min(str.size(), stop.size());
        const char last_char = str.back();
        for (size_t len = max_len; len > 0; --len) {
            if (stop[len - 1] == last_char) {
                if (string_ends_with(str, stop.substr(0, len))) {
                    return str.size() - len;
                }
            }
        }
    }
    return std::string::npos;
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides);

// parse a keep_alive style duration ("5m", "1h30m", "0", "-1s") into milliseconds; negative -> -1 (forever)
// bare_number_is_seconds also accepts a plain integer as seconds, as OLLAMA_KEEP_ALIVE does
bool string_parse_duration_ms(const std::string & str, int64_t & out_ms, bool bare_number_is_seconds = false);
void string_process_escapes(std::string & input);
std::string string_unescape(const std::string& str);

std::vector<std::string> string_extract(const std::string& str, const char c, std::vector<size_t>& posi);
void argparse_expiring_logit_bias(const std::string & content, common_params_sampling & sparams);

bool string_is_found(const std::string& window, const std::string& str, size_t& pos);

std::string string_from(bool value);
std::string string_from(const std::vector<int> & values);
std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens);

bool glob_match(const std::string & pattern, const std::string & str);

//
// Unicode utils
//

#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string & str);
std::string  wstring_to_utf8(const std::wstring & str);
#endif

//
// Environment utils
//

// portable environment access, an unset variable reads as an empty string
// and setting an empty value unsets the variable
std::string common_get_env(const std::string & name);
void        common_set_env(const std::string & name, const std::string & value);
void string_assign_append(std::string& dst, const std::string_view& sv, const std::string& str, const int32_t pos);

//
// Filesystem utils
//

bool fs_validate_filename(const std::string & filename, bool allow_subdirs = false);
bool fs_is_directory(const std::string & path);
bool fs_create_directory_with_parents(const std::string & path); // common.cpp (declaration restored 2026-11-05)

// some old libstdc++ versions don't follow symlinks here, so adding a trailing "/" fixes it: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=101510
inline bool common_create_directories(const std::filesystem::path & path, std::error_code & ec) {
#if defined(__linux__)
    return std::filesystem::create_directories(path / "", ec);
#else
    return std::filesystem::create_directories(path, ec);
#endif
}

std::filesystem::path fs_get_cache_directory();
std::filesystem::path fs_get_cache_file(const std::string & filename);
std::filesystem::path fs_get_config_directory();

void fs_write_atomic(const std::filesystem::path & path, const std::string & data);

//
// TTY utils
//

// Auto-detect if colors can be enabled based on terminal and environment
bool tty_can_use_colors();
bool tty_enable_ansi(); // false when stdout or stderr is a console that cannot render ANSI sequences

// Check if the given file is attached to a terminal
bool common_is_tty(FILE * file);


//
// Model utils
//

struct common_sampler;

// typed decision models, see "<arch>.decision.type" in the model metadata
enum common_decision_type {
    COMMON_DECISION_TYPE_NONE,    // not a decision model
    COMMON_DECISION_TYPE_OPENJEV, // logits of one label token per option, read at the last prompt token
    COMMON_DECISION_TYPE_LEV,     // same as openjev, noul is read from a rating scale
    COMMON_DECISION_TYPE_KEV,     // dot product of the hidden states of the last token and of one end token per option
    COMMON_DECISION_TYPE_NIMBLE,  // same as openjev, the prompt lists all the questions of the request
    COMMON_DECISION_TYPE_LAYA,    // score of one marker token per option, read from the embeddings output
    COMMON_DECISION_TYPE_CLEF,    // all questions in one prompt, score of option i read from the embeddings output at row i
    COMMON_DECISION_TYPE_PPLX_DECIDER, // same as openjev, label codes of 1 or 2 letters
    COMMON_DECISION_TYPE_VLLM_DECISION2, // score of each option read from the embeddings output, the options of a question are marked in the batch
    COMMON_DECISION_TYPE_VLLM_D3, // same as pplx-decider, non-causal attention
    COMMON_DECISION_TYPE_LFM2_D1, // same as openjev, the labels depend on the question type
    COMMON_DECISION_TYPE_LFM2_D1_OMNI, // same as laya, other prompt layout
    COMMON_DECISION_TYPE_GRANITE_SWITCH, // same as openjev, run by the decision aLoRA of granite switch
    COMMON_DECISION_TYPE_H2O_LIGHTNING, // same as openjev, the labels are read after an answer prefill
    COMMON_DECISION_TYPE_UNKNOWN, // a decision model of a type that is not supported
};

common_decision_type common_get_decision_type(const struct llama_model * model);

// metadata of a GGUF file, read without loading the model
struct common_gguf_info {
    common_decision_type decision_type = COMMON_DECISION_TYPE_UNKNOWN; // UNKNOWN if the file is missing, unreadable, or invalid
    uint32_t             n_ctx_train   = 0;                            // 0 if unknown
    bool                 has_chat_tmpl = false;                        // has the default tokenizer.chat_template
};

common_gguf_info common_get_gguf_info(const std::string & fname);

// note: defines the model, context, samplers, ets. lifetimes
struct common_init_result {
    common_init_result(common_params & params, bool model_only = false);
    ~common_init_result();

    llama_model * model();
    llama_context * context();

    common_sampler * sampler(llama_seq_id seq_id);
    void reset_samplers();

    std::vector<llama_adapter_lora_ptr> & lora();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_init_result_ptr = std::unique_ptr<common_init_result>;

// --cram-probe: logs and returns the read-only sysfs probe result (include src/llama-cram-probe.h to use the result)
llama_cram_probe_result common_cram_probe_log(const char * sysfs_root = nullptr);

// --vram-spill: pushes params.vram_spill_* into ggml (ggml_vram_spill_configure) and logs one always-on line saying what was
// configured and whether this build can act on it. Returns true when the policy is active (mode != off). Called at the
// top of common_init_from_params, before any weight buffer is allocated.
bool common_vram_spill_apply(const common_params & params);

// --model-checksum: streams the file at `path` through SHA-256 and compares with `expected_hex` (case-insensitive).
// Returns true on match. On false, `err` names the expected and the actual digest (or says the file cannot be read /
// the expected value is not 64 hex characters). Row S69-ENGREV-SGLANG-MODEL-CHECKSUM-2026-10-27-01.
bool common_verify_model_checksum(const std::string & path, const std::string & expected_hex, std::string & err);

common_init_result_ptr common_init_from_params(common_params & params, bool model_only = false);

struct llama_model_params   common_model_params_to_llama  (      common_params & params);
struct llama_context_params common_context_params_to_llama(const common_params & params);

// clear LoRA adapters from context, then apply new list of adapters
void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora);

// model endpoint from env
std::string common_get_model_endpoint();

// for testing purposes
char * common_get_model_or_exit(int, char*[]);

//
// Threadpool utils
//

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params);

struct common_threadpools {
    common_threadpools() = default;
    ~common_threadpools();

    common_threadpools(const common_threadpools &) = delete;
    common_threadpools & operator=(const common_threadpools &) = delete;

    void init(llama_context * ctx, const common_params & params);

private:
    ggml_threadpool * threadpool       = nullptr;
    ggml_threadpool * threadpool_batch = nullptr;

    decltype(ggml_threadpool_free) * free_fn = nullptr;
};

//
// Context utils
//

enum common_context_seq_rm_type {
    COMMON_CONTEXT_SEQ_RM_TYPE_NO           = 0, // seq_rm not supported (e.g. no memory module)
    COMMON_CONTEXT_SEQ_RM_TYPE_PART         = 1, // can seq_rm partial sequences
    COMMON_CONTEXT_SEQ_RM_TYPE_FULL         = 2, // can seq_rm full sequences only
    COMMON_CONTEXT_SEQ_RM_TYPE_RS = 3, // can seq_rm partial sequences, bounded by n_rs_seq
};

// check if the llama_context can remove sequences
// note: clears the memory of the context
common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx);

struct common_memory {
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    void init(llama_context * ctx_tgt, llama_context * ctx_dft = nullptr);

    // aborts execution on failure
    void seq_rm (llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) const;
    void seq_cp (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) const;
};

//
// Batch utils
//

// wrapper around llama_batch_ext that provide getter functions for downstream code
// entries can exceed n_batch, use get_sub_batch() to decode them in chunks
struct common_batch {
    struct token {
        llama_token  id;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos; // only pos[0] is used for text tokens
        llama_seq_id seq_id; // the first sequence id, see add_seq()
        bool         output;
        llama_embd   embd; // non-owning view of the data passed to add_embd()/set_embd(), data == NULL if none
        llama_embd   state; // non-owning view of the data passed to set_embd_state(), data == NULL if none
        std::vector<llama_seq_id> seq_ids_extra; // see add_seq()
        int32_t      decision_order = 0; // see llama_batch_ext_set_decision_order()
    };

    std::vector<token> tokens; // mirror of the entries, tokens[i] describes batch index i
    llama_batch_ext_ptr batch;

    int32_t n_pos = 1; // positions per embedding entry, GGML_MROPE_SECTIONS for MROPE/IMROPE

    common_batch() = default;
    common_batch(struct llama_context * ctx);

    llama_batch_ext * get() { return get_sub_batch(0, size()); }

    // render entries [off, off + n) into batch, the result is overwritten by the next call
    llama_batch_ext * get_sub_batch(int32_t off, int32_t n);

    // content type of the batch, all entries carry the same combination
    bool has_token() const { return !tokens.empty() && tokens[0].id != LLAMA_TOKEN_NULL; }
    bool has_embd () const { return !tokens.empty() && tokens[0].embd.data != nullptr; }

    void clear();

    // returns the batch index
    int32_t add(llama_token id, llama_pos pos, llama_seq_id seq_id, bool output);

    // same, with the entry shared by all seq_ids (must not be empty)
    int32_t add(llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool output);

    // add the entry at idx to another sequence, tokens[idx].seq_id keeps the first one
    bool add_seq(int32_t idx, llama_seq_id seq_id);

    bool set_output(int32_t idx, bool value);

    // attach a token embedding to the entry at idx, can only be set once per entry
    bool set_embd(int32_t idx, llama_embd embd);

    // attach a state embedding (e.g. the target hidden state for MTP) to the entry at idx, can only be set once per entry
    bool set_embd_state(int32_t idx, llama_embd state);

    // add an embedding-only entry (no token id)
    // pos points to n_pos positions
    int32_t add_embd(llama_embd embd, const llama_pos * pos, llama_seq_id seq_id, bool output);

    int32_t size() const { return (int32_t) tokens.size(); }
};

// create a single-sequence batch from a list of tokens
// positions continue from the memory, last token always have output_logits set to true
common_batch common_batch_get_one(struct llama_context * ctx, const llama_token * tokens, int32_t n_tokens);
common_batch common_batch_get_one(struct llama_context * ctx, const llama_tokens & tokens);

// fork llama_batch helpers (llama_batch_init + llama_decode callers); kept next to common_batch, which wraps
// llama_batch_ext for llama_process. Restored 2026-11-04: the upstream merge had deleted them under 29 callers.
void common_batch_clear(struct llama_batch & batch);

void common_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits);

// decodes a single batch of tokens for a prompt and manages session tokens
//
// Note: We save state before the last token so that we can replay it to ensure
// compatibility with all memory types. Recurrent/hybrid models cannot remove
// tokens from memory, so this approach works across all model architectures.
bool common_prompt_batch_decode(
              struct llama_context * ctx,
                const llama_tokens & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state);

// replays the last token after loading state to regenerate logits
// used after loading session state to ensure the sampling context has valid logits
bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos);

//
// Vocab utils
//

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

// REMOVED 2026-09-13: a THIRD common_tokenize declaration stood here, taking
// `const struct llama_vocab* vocab, const std::string& text, bool, bool` -- the SAME signature as
// the vocab overload immediately above, differing only in where the `*` and `&` sit against the
// type name, which C++ does not distinguish. Redeclaring it while repeating the default argument is
// the error:
//     common.h:1292: error: default argument given for parameter 4 of
//                    'std::vector<int> common_tokenize(const llama_vocab*, const std::string&,
//                    bool, bool)' [-fpermissive]
// The ctx overload and the vocab overload above are genuinely different and both stay.

std::vector<llama_token> llama_tokenize(
    const struct llama_vocab * vocab,
    const std::string & text,
    bool   add_special,
    bool   parse_special = false);

// tokenizes a token into a piece, optionally renders special/control tokens
// should work similar to Python's `tokenizer.id_to_piece`
std::string common_token_to_piece(
        const struct llama_context * ctx,
                       llama_token   token,
                       bool          special = true);

std::string common_token_to_piece(
          const struct llama_vocab * vocab,
                       llama_token   token,
                       bool          special = true);

// the byte table llguidance wants for a vocab: lens[i] is the number of bytes token i contributes MID-SEQUENCE and bytes holds
// all of them back to back (a special token carries a leading 0xff marker). Sized by what the vocab needs, not by a guess.
void common_vocab_token_table(const llama_vocab * vocab, std::vector<uint32_t> & lens, std::vector<uint8_t> & bytes);

// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
// optionally renders special/control tokens
std::string common_detokenize(
        const llama_context * ctx,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

std::string common_detokenize(
    const struct llama_vocab * vocab,
    const std::vector<llama_token> & tokens,
                                    bool special = true);

// REMOVED 2026-09-13: a THIRD common_token_to_piece declaration stood here with the SAME signature
// as the vocab overload above -- `const struct llama_vocab *, llama_token, bool` -- differing only
// in indentation. Repeating the default argument on a redeclaration is the error:
//     common.h:1329: error: default argument given for parameter 3 of
//                    'std::string common_token_to_piece(const llama_vocab*, llama_token, bool)'
// The ctx overload and the vocab overload above are genuinely different and both stay. This is the
// same merge-doubling as string_split and common_tokenize in this file.

// Uses the value from the model metadata if possible, otherwise
// defaults to true when model type is SPM, otherwise false.
bool llama_should_add_bos_token(const llama_model * model);




//
// KV cache utils
//

// RESEARCH-DONE: git show a4090d1174aed22dde5cacce2a4c27656b987a2f (full commit read).
// RESTORING a capability here, not disabling one (S69-EXAMPLES-KV-VIEW-DISPOSITION-2026-09-08-01):
// these 2 functions were DECLARED here (un-prefixed, fork-native spelling -- confirmed
// distinct from upstream's own common_-prefixed common_kv_cache_dump_view_seqs, which
// upstream itself later deleted entirely in that commit, "llama : remove
// llama_kv_cache_view API + remove deprecated (#13653)") but had NO DEFINITION anywhere in
// the currently-staged common/common.cpp -- a declaration-survived-definition-dropped bug
// (the exact failure mode flagged as most frequent in this merge), caught because
// examples/parallel/parallel.cpp (staged in this same dispatch) calls
// llama_kv_cache_dump_view_seqs directly and would otherwise link-fail. Recovered the exact
// fork-native body from common/common.cpp.pre-fmt-all (a pre-formatting snapshot of this same
// file that still had it, lines 6337-6405 there) rather than reinventing it. Made `inline`
// (not a plain free function) since common.h is included by many translation units
// (common.cpp, arg.cpp, chat.cpp, the 3 examples, ...) -- a non-inline definition here would
// be an ODR/multiple-definition link error the moment more than one TU includes this header,
// which every one of them does.
//
// Dump the KV cache view with the number of sequences per cell.
inline void llama_kv_cache_dump_view(const llama_kv_cache_view & view, int row_size = 80) {
    static const char slot_chars[] = ".123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+";

    printf("=== Dumping KV cache. total cells %d, max sequences per cell %d, populated cells %d, total tokens in cache %d, largest empty slot=%d @ %d",
        view.n_cells, view.n_seq_max, view.used_cells, view.token_count, view.max_contiguous, view.max_contiguous_idx);

    llama_kv_cache_view_cell * c_curr = view.cells;
    llama_seq_id * cs_curr = view.cells_sequences;

    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        if (i % row_size == 0) {
            printf("\n%5d: ", i);
        }
        int seq_count = 0;
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] >= 0) { seq_count++; }
        }
        putchar(slot_chars[std::min(sizeof(slot_chars) - 2, size_t(seq_count))]);
    }

    printf("\n=== Done dumping\n");
}

// Dump the KV cache view showing individual sequences in each cell (long output).
inline void llama_kv_cache_dump_view_seqs(const llama_kv_cache_view & view, int row_size = 40) {
    static const char slot_chars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    printf("=== Dumping KV cache. total cells %d, max sequences per cell %d, populated cells %d, total tokens in cache %d, largest empty slot=%d @ %d\n",
        view.n_cells, view.n_seq_max, view.used_cells, view.token_count, view.max_contiguous, view.max_contiguous_idx);

    std::unordered_map<llama_seq_id, size_t> seqs;
    llama_kv_cache_view_cell * c_curr = view.cells;
    llama_seq_id * cs_curr = view.cells_sequences;

    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] < 0) { continue; }
            if (seqs.find(cs_curr[j]) == seqs.end()) {
                if (seqs.size() + 1 >= sizeof(slot_chars)) { break; }
                const size_t sz = seqs.size();
                seqs[cs_curr[j]] = sz;
            }
        }
        if (seqs.size() + 1 >= sizeof(slot_chars)) { break; }
    }

    printf("=== Sequence legend: ");
    for (const auto & it : seqs) {
        printf("%zu=%d, ", it.second, it.first);
    }
    printf("'+'=other sequence ids");

    c_curr = view.cells;
    cs_curr = view.cells_sequences;
    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        if (i % row_size == 0) {
            printf("\n%5d: ", i);
        }
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] >= 0) {
                const auto & it = seqs.find(cs_curr[j]);
                putchar(it != seqs.end() ? int(slot_chars[it->second]) : '+');
            } else {
                putchar('.');
            }
        }
        putchar(' ');
    }

    printf("\n=== Done dumping\n");
}

//
// Embedding utils
//

void common_embd_normalize(const float * inp, float * out, int n, int embd_norm = 2);

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n);

//
// Control vector utils
//

struct common_control_vector_data {
    int n_embd;

    // stores data for layers [1, n_layer] where n_layer = data.size() / n_embd
    std::vector<float> data;
};

struct common_control_vector_load_info {
    float strength;

    std::string fname;
};

// Load control vectors, scale each by strength, and add them together.
// Direction layer indices must be in [1, n_layer].
// On error, returns {-1, empty}
common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos, int32_t n_layer);

//
// Split utils
//

namespace {

const char * const LLM_KV_SPLIT_NO            = "split.no";
const char * const LLM_KV_SPLIT_COUNT         = "split.count";
const char * const LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count";

}

//
// FFN offload utils
//

const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

const char * const LLM_FFN_DENSE_REGEX = "\\.ffn_(up|down|gate)\\.";

inline std::string llm_ffn_exps_block_regex(int idx) {
    return string_format("blk\\.%d%s", idx, LLM_FFN_EXPS_REGEX);
}

inline std::string llm_ffn_block_regex(int idx, const char * ffn_regex) {
    return string_format("blk\\.%d%s", idx, ffn_regex);
}

inline llama_model_tensor_buft_override llm_ffn_exps_cpu_override() {
    return { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() };
}

inline void llm_add_n_cpu_ffn_overrides(int n, const char * ffn_regex, std::vector<llama_model_tensor_buft_override> & overrides) {
    // keep strings alive and avoid leaking memory by storing them in a static list
    static std::list<std::string> buft_override_strings;
    for (int i = 0; i < n; ++i) {
        buft_override_strings.push_back(llm_ffn_block_regex(i, ffn_regex));
        overrides.push_back({buft_override_strings.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
}

//
// training utils
//

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride);

// llama_opt_params::get_opt_pars callback: userdata is a `lr_opt *`; defined in common.cpp (was undeclared, so
// examples/training/finetune.cpp could not name it)
struct ggml_opt_optimizer_params common_opt_lr_pars(void * userdata);

// "adamw" or "sgd" (case insensitive)
enum ggml_opt_optimizer_type common_opt_get_optimizer(const char *);

//
// prompt utils
//

class llama_cram_soft_store; // src/llama-cram-soft.h

// Process-wide counters of the CRAM software tier (--ctx-ckpt-cram-soft); they show that the opt-in path ran.
struct common_ckpt_cram_counters {
    uint64_t stores = 0;      // checkpoints compressed into a store
    uint64_t reloads = 0;     // checkpoints read back out of a store
    uint64_t raw_bytes = 0;   // sum of the uncompressed sizes stored
    uint64_t stored_bytes = 0;// sum of payload + block-table bytes
    uint64_t read_bytes = 0;  // bytes decoded by reloads
};
common_ckpt_cram_counters common_ckpt_cram_get_counters();
void                      common_ckpt_cram_reset_counters();

struct common_prompt_checkpoint {
    int64_t n_tokens;

    // (optional) id of the task that created the checkpoint
    int id_task = -1;

    // --ctx-ckpt-dominance bookkeeping: anchor = taken at the prompt end, branch = taken at a branch point (LCP snapshot)
    bool anchor = false;
    bool branch = false;

    llama_pos pos_min;
    llama_pos pos_max;

    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;

    // (optional) speculative-decoding implementation state stashed with the checkpoint
    // (e.g. eagle3's deferred-boundary g_embd row)
    std::vector<uint8_t> data_spec;

    // Disk spill (--ctx-ckpt-spill-dir). A spilled checkpoint keeps its positions in memory but its three data
    // vectors live in this file; reload() brings them back. Empty = never spilled.
    // Reimplemented from the idea of ik_llama.cpp PR 2446 (MIT, source read); the file format is our own.
    std::string spill_path;

    // NVMe tier (--cache-disk-dir, common_disk_tier). When set, the three data vectors are (or were) written to this blob.
    // Every copy of the checkpoint shares it (counted reference): the blob is erased from the tier when the last copy goes,
    // so a copy can never point at a file another copy deleted. Row S69-TIERED-CKPT-BLOBS-INTO-DISK-TIER-2026-10-05-01.
    std::shared_ptr<common_disk_blob> disk;

    // CRAM software tier (--ctx-ckpt-cram-soft). The three data vectors, packed back to back, live in a compressed read-only
    // block store; the vectors are freed. Lengths of the three records are in cram_len. The store is immutable, so copies of the
    // checkpoint share it. reload() reads it back (and keeps the store, so compressing again is free).
    std::shared_ptr<llama_cram_soft_store> cram;
    uint64_t cram_len[3] = {0, 0, 0};

    size_t size() const;

    bool empty() const;
    void clear();

    // data is on disk (spill file or tier blob) and not in memory
    bool spilled() const { return data_tgt.empty() && (!spill_path.empty() || disk != nullptr || cram != nullptr); }
    // compress the resident data into a llama_cram_soft_store and free the vectors. False (data stays resident) if nothing is resident.
    bool cram_compress();
    // write the data to `tier` and free the in-memory vectors; if a blob of this checkpoint is already in `tier` (it was reloaded
    // earlier) nothing is written again, the memory is just freed. False (data stays resident) when there is nothing resident
    // or the tier refuses (size cap, write budget, I/O error).
    bool spill_tier(const std::shared_ptr<common_disk_tier> & tier);
    // payload bytes a spill_tier() would write for the target vector / the packed draft+spec vector (to ask the tier first)
    size_t tier_payload_a() const { return data_tgt.size(); }
    size_t tier_payload_b() const { return 2 * sizeof(uint64_t) + data_dft.size() + data_spec.size(); }
    // identifies this checkpoint to the tier: a blob that carries another tag is never handed back
    int64_t tier_tag() const;
    // write the data to `path`, verify the size on disk, then free the in-memory vectors. False (and the data stays
    // resident, no file left behind) on any write problem or when there is nothing resident to offload.
    bool spill(const std::string & path);
    // read a spilled checkpoint back (from the spill file or from the tier blob). False if it is not spilled or the file/blob is
    // missing, stale or corrupt (the file is then deleted / the blob dropped and the reference cleared, so the caller falls back to
    // reprocessing the prompt). A tier blob stays in the tier after a successful reload, so spilling the same checkpoint again
    // costs no write.
    bool reload();
    // the file half of reload(); remove_on_fail = false leaves a bad file in place (it is not ours to delete)
    bool reload_file(bool remove_on_fail);
    // For a COPY of a slot's checkpoint that --ctx-ckpt-spill-dir wrote to a file (server_prompt_cache::alloc with the NVMe tier on):
    // read the data into this copy and let go of the path, so the copy never shares the slot's file. False (path cleared, file left
    // alone) when the file cannot be read; the caller drops the copy.
    bool adopt_spill_file();
    // delete the spill file, if any
    void remove_spill_file();

    void update_pos(
            int64_t n_tokens,
            llama_pos pos_min,
            llama_pos pos_max);

    void update_tgt(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags);

    void update_dft(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags);

    // return false if the state could not be restored
    bool load_tgt(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags) const;

    bool load_dft(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags) const;

    void clear_tgt();
    void clear_dft();
};
