#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include "build-info.h"
#include "common.h"
#include "../src/llama-cram-soft.h"
#include "../src/llama-cram-probe.h"
#include "ggml-vram-spill.h"
#include "ggml-backend.h"
#include <mutex>

#include "../src/llama-ext.h"

#include "fit.h"
#include "download.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
#include <sys/types.h>
#include <sys/stat.h>
#include <pwd.h>
#endif

#if defined(_AIX)
#include <sys/systemcfg.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

common_time_meas::common_time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

common_time_meas::~common_time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

// --- FORK-ONLY, carried verbatim from ours in this merge (declared in the already-
//     resolved common.h; upstream/theirs has no equivalent) ---
bool common_speculative_type_is_dflash_family(enum common_speculative_type type) {
    return type == COMMON_SPECULATIVE_TYPE_DFLASH || type == COMMON_SPECULATIVE_TYPE_DSPARK;
}

bool common_speculative_type_draws_from_q(enum common_speculative_type type) {
    return type == COMMON_SPECULATIVE_TYPE_DRAFT || type == COMMON_SPECULATIVE_TYPE_MTP;
}

bool common_speculative_type_uses_target_features(enum common_speculative_type type) {
    return type == COMMON_SPECULATIVE_TYPE_MTP || common_speculative_type_is_dflash_family(type);
}

bool common_speculative_type_is_self_spec(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
        case COMMON_SPECULATIVE_TYPE_SUFFIX:
        case COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON:
            return true;
        default:
            return false;
    }
}

static int32_t common_speculative_stage_effective_n_max(
        const common_params_speculative & params,
        const common_speculative_stage_params & stage) {
    return stage.has_n_max_override() ? stage.n_max : params.n_max;
}

static int32_t common_speculative_stage_effective_n_min(
        const common_params_speculative & params,
        const common_speculative_stage_params & stage) {
    return stage.has_n_min_override() ? stage.n_min : params.n_min;
}

std::vector<common_speculative_stage_params> common_params_speculative::get_resolved_stages() const {
    if (!stages.empty()) {
        std::vector<common_speculative_stage_params> resolved;
        resolved.reserve(stages.size());

        for (const auto & stage : stages) {
            if (stage.type != COMMON_SPECULATIVE_TYPE_NONE) {
                resolved.push_back(stage);
            }
        }

        return resolved;
    }

    if (type == COMMON_SPECULATIVE_TYPE_NONE) {
        return {};
    }

    return {{ .type = type }};
}

common_params_speculative common_params_speculative::with_stage_overrides(const common_speculative_stage_params & stage) const {
    common_params_speculative result = *this;

    result.type = stage.type;

    if (stage.has_n_max_override()) {
        result.n_max = stage.n_max;
    }
    if (stage.has_n_min_override()) {
        result.n_min = stage.n_min;
    }
    if (stage.has_p_min_override()) {
        result.p_min = stage.p_min;
    }
    if (stage.has_mtp_heads_override()) {
        result.mtp_heads = stage.mtp_heads;
    }
    if (stage.has_ngram_size_n_override()) {
        result.ngram_size_n = stage.ngram_size_n;
        result.ngram_mod.reset();
    }
    if (stage.has_ngram_size_m_override()) {
        result.ngram_size_m = stage.ngram_size_m;
    }
    if (stage.has_ngram_min_hits_override()) {
        result.ngram_min_hits = stage.ngram_min_hits;
    }
    if (stage.has_suffix_min_match_len_override()) {
        result.suffix_min_match_len = stage.suffix_min_match_len;
    }
    if (stage.has_suffix_max_depth_override()) {
        result.suffix_max_depth = stage.suffix_max_depth;
    }
    if (stage.has_suffix_max_nodes_override()) {
        result.suffix_max_nodes = stage.suffix_max_nodes;
    }
    if (stage.has_suffix_corpus_override()) {
        result.suffix_corpus = stage.suffix_corpus;
    }

    result.n_max = std::max(result.n_max, 0);
    result.n_min = std::max(0, std::min(result.n_min, result.n_max));
    result.mtp_heads = std::max(result.mtp_heads, 0);
    result.stages.clear();

    return result;
}

bool common_params_speculative::has_stage_chain() const {
    return !get_resolved_stages().empty();
}

bool common_params_speculative::has_stage_type(common_speculative_type stage_type) const {
    const auto resolved = get_resolved_stages();
    return std::any_of(resolved.begin(), resolved.end(), [stage_type](const common_speculative_stage_params & stage) {
        return stage.type == stage_type;
    });
}

bool common_params_speculative::has_dflash_family_stage() const {
    const auto resolved = get_resolved_stages();
    return std::any_of(resolved.begin(), resolved.end(), [](const common_speculative_stage_params & stage) {
        return common_speculative_type_is_dflash_family(stage.type);
    });
}

bool common_params_speculative::uses_target_features() const {
    const auto resolved = get_resolved_stages();
    return std::any_of(resolved.begin(), resolved.end(), [](const common_speculative_stage_params & stage) {
        return common_speculative_type_uses_target_features(stage.type);
    });
}

void common_params_speculative::remove_stage_type(common_speculative_type stage_type) {
    stages.erase(std::remove_if(stages.begin(), stages.end(), [stage_type](const common_speculative_stage_params & stage) {
        return stage.type == stage_type;
    }), stages.end());

    if (type == stage_type) {
        const auto resolved = get_resolved_stages();
        type = resolved.empty() ? COMMON_SPECULATIVE_TYPE_NONE : resolved.front().type;
    }
}

bool common_params_speculative::has_composite_stage_chain() const {
    return get_resolved_stages().size() > 1;
}

bool common_params_speculative::needs_dft_model() const {
    return has_stage_type(COMMON_SPECULATIVE_TYPE_DRAFT) ||
        has_dflash_family_stage() ||
        has_stage_type(COMMON_SPECULATIVE_TYPE_EAGLE3) ||   // S69 #33: eagle3 needs a draft-head model (-md)
        (has_stage_type(COMMON_SPECULATIVE_TYPE_MTP) && has_dft());
}

void common_params_speculative::clear_dft() {
    if (model_dft != nullptr) {
        llama_free_model(model_dft);
        model_dft = nullptr;
    }

    model.clear();
    params.clear();
    mparams_dft.path.clear();
    draft.mparams = common_params_model();
    cparams_dft = llama_context_default_params();
}

void common_params_speculative_fold_draft(common_params_speculative & sp) {
    const common_params_speculative_draft def;
    const common_params_speculative_draft & d = sp.draft;
    auto cpu_differs = [](const common_cpu_params & a, const common_cpu_params & b) {
        if (a.n_threads != b.n_threads || a.mask_valid != b.mask_valid || a.priority != b.priority || a.strict_cpu != b.strict_cpu ||
            a.poll != b.poll || a.place_physical != b.place_physical) {
            return true;
        }
        for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
            if (a.cpumask[i] != b.cpumask[i]) {
                return true;
            }
        }
        return false;
    };
    if (d.n_max   != def.n_max)   { sp.n_max   = d.n_max; }
    if (d.n_min   != def.n_min)   { sp.n_min   = d.n_min; }
    if (d.p_split != def.p_split) { sp.p_split = d.p_split; }
    if (d.p_min   != def.p_min)   { sp.p_min   = d.p_min; sp.p_min_explicit = true; }
    if (d.backend_sampling != def.backend_sampling) { sp.backend_sampling = d.backend_sampling; }
    if (d.probabilistic    != def.probabilistic)    { sp.draft_sample     = d.probabilistic; }
    if (!d.mparams.empty() || !d.mparams.url.empty()) {
        sp.mparams_dft = d.mparams;
        sp.model       = d.mparams.path; // dft_path() reads `model` first
    }
    if (d.ctx_tgt != nullptr) { sp.ctx_tgt = d.ctx_tgt; }
    if (d.ctx_dft != nullptr) { sp.ctx_dft = d.ctx_dft; }
    if (d.n_gpu_layers != def.n_gpu_layers) { sp.n_gpu_layers = d.n_gpu_layers; }
    if (d.n_ubatch     != def.n_ubatch)     { sp.n_ubatch     = d.n_ubatch; }
    if (d.cache_type_k != def.cache_type_k) { sp.cache_type_k = ggml_type_name(d.cache_type_k); }
    if (d.cache_type_v != def.cache_type_v) { sp.cache_type_v = ggml_type_name(d.cache_type_v); }
    if (cpu_differs(d.cpuparams,       def.cpuparams))       { sp.cpuparams       = d.cpuparams; }
    if (cpu_differs(d.cpuparams_batch, def.cpuparams_batch)) { sp.cpuparams_batch = d.cpuparams_batch; }
    if (!d.devices.empty()) {
        std::string names;
        for (ggml_backend_dev_t dev : d.devices) {
            if (dev == nullptr) {
                continue;
            }
            names += (names.empty() ? "" : ",") + std::string(ggml_backend_dev_name(dev));
        }
        sp.devices = names;
    }
    if (!d.tensor_buft_overrides.empty()) {
        sp.tensor_buft_overrides = d.tensor_buft_overrides;
        if (sp.tensor_buft_overrides.back().pattern != nullptr) {
            sp.tensor_buft_overrides.push_back({nullptr, nullptr});
        }
    }
    const enum llama_lazy_mode lazy = d.lazy_mode;
    sp.draft = common_params_speculative_draft();
    sp.draft.lazy_mode = lazy;
}

int32_t common_params_speculative::get_max_stage_n_max() const {
    const auto resolved = get_resolved_stages();
    if (resolved.empty()) {
        return std::max(n_max, 0);
    }

    int32_t max_n_max = 0;
    for (const auto & stage : resolved) {
        max_n_max = std::max(max_n_max, common_speculative_stage_effective_n_max(*this, stage));
    }

    return std::max(max_n_max, 0);
}

int32_t common_params_speculative::get_min_usable_stage_n_min() const {
    const auto resolved = get_resolved_stages();
    if (resolved.empty()) {
        return std::max(0, std::min(n_min, n_max));
    }

    int32_t min_n_min = INT_MAX;
    for (const auto & stage : resolved) {
        min_n_min = std::min(min_n_min, std::max(0, std::min(common_speculative_stage_effective_n_min(*this, stage), common_speculative_stage_effective_n_max(*this, stage))));
    }

    return min_n_min == INT_MAX ? 0 : min_n_min;
}

bool common_speculative_validate_chain(const common_params_speculative & params, std::string * error) {
    const auto fail = [error](const std::string & msg) {
        if (error != nullptr) {
            *error = msg;
        }
        return false;
    };

    const auto resolved = params.get_resolved_stages();
    if (resolved.empty()) {
        return true;
    }

    if (resolved.size() > 2) {
        return fail("at most two speculative stages are supported in this PR");
    }

    std::unordered_set<int> seen_types;
    for (const auto & stage : resolved) {
        if (stage.type == COMMON_SPECULATIVE_TYPE_NONE && resolved.size() > 1) {
            return fail("the 'none' speculative stage cannot be combined with other stages");
        }

        if (!seen_types.insert((int) stage.type).second) {
            return fail("duplicate speculative stage type in chain: " + common_speculative_type_to_str(stage.type));
        }

        const auto stage_params = params.with_stage_overrides(stage);
        if (stage_params.n_min > stage_params.n_max) {
            return fail("speculative stage has n_min greater than n_max");
        }

        if ((stage.type == COMMON_SPECULATIVE_TYPE_DRAFT || common_speculative_type_is_dflash_family(stage.type)) && !params.has_dft()) {
            return fail(common_speculative_type_to_str(stage.type) + " speculative stage requires a draft model (-md/--model-draft)");
        }

    }

    if (resolved.size() == 2) {
        const auto first = resolved[0].type;
        const auto second = resolved[1].type;

        if (!common_speculative_type_is_self_spec(first)) {
            return fail("two-stage speculative mode currently requires a self-spec stage first");
        }

        if (second != COMMON_SPECULATIVE_TYPE_MTP && second != COMMON_SPECULATIVE_TYPE_DRAFT) {
            return fail("two-stage speculative mode currently supports only MTP or draft-model fallback after self-spec");
        }
    }

    return true;
}

std::string common_speculative_stage_chain_to_str(const common_params_speculative & params) {
    const auto resolved = params.get_resolved_stages();
    if (resolved.empty()) {
        return "none";
    }

    std::ostringstream oss;
    for (size_t i = 0; i < resolved.size(); ++i) {
        if (i > 0) {
            oss << " -> ";
        }
        oss << common_speculative_type_to_str(resolved[i].type);
    }

    return oss.str();
}

//
// CPU utils
//

#ifdef __linux__
static std::string cpu_physical_core_key(int cpu) {
    std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
        + std::to_string(cpu) + "/topology/thread_siblings_list");
    std::string line;
    if (!std::getline(thread_siblings, line)) {
        return {};
    }
    return line;
}
#endif

int32_t common_cpu_get_num_physical_cores() {
#if defined(_AIX)
    int32_t logical_cpus = _system_configuration.ncpus;
    int32_t smt_threads = _system_configuration.smt_threads;
    if (smt_threads > 0) {
        return static_cast<int32_t>(logical_cpus / smt_threads);
    }
    if (logical_cpus > 0) {
        return static_cast<int32_t>(logical_cpus);
    }
#elif defined(__linux__)
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    const long n_cpu_conf = sysconf(_SC_NPROCESSORS_CONF);
    const int  n_cpu      = n_cpu_conf > 0 ? (int) std::min((long) CPU_SETSIZE, n_cpu_conf) : (int) CPU_SETSIZE;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        const std::string key = cpu_physical_core_key(cpu);
        if (!key.empty()) { // skip cpus with no topology (offline / holes)
            siblings.insert(key);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !(defined(__MINGW64__) && defined(__aarch64__)) // windows 7 and later
    // Upstream excluded every MinGW build here (its open item was windows + arm64 + mingw64). x86_64 MinGW-w64 declares
    // GetLogicalProcessorInformationEx (sysinfoapi.h, _WIN32_WINNT >= 0x0601, its default), and excluding it made MinGW
    // builds guess hardware_concurrency()/2 instead of counting cores. Only the arm64 MinGW case stays excluded.
    unsigned int n_threads_win = std::thread::hardware_concurrency();
    unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return default_threads;
        }
    }

    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
        return default_threads;
    }

    int32_t num_physical_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            num_physical_cores += info->Processor.GroupCount;
        }
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char*>(info) + info->Size);
    }

    return num_physical_cores > 0 ? num_physical_cores : default_threads;
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

// --cpu-placement physical. Behaviour from Thireus/ik_llama afdbc5e13 (#2565, MIT): its --cpu-affinity mode orders the hybrid
// P-cores one logical CPU per physical core, telling siblings apart by thread_siblings_list (SOURCE-READ common/common.cpp
// @afdbc5e13: cpu_physical_core_key, cpu_detect_math_cpus). Here it covers every physical core of any x86/ARM CPU, has a
// Windows arm, and feeds the existing threadpool cpumask (-C/-Cr) with strict placement instead of a second pinning list.
// No code copied from it beyond the sysfs path, which is the kernel ABI.
std::vector<std::vector<int32_t>> common_cpu_core_siblings() {
    std::vector<std::vector<int32_t>> cores;
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !(defined(__MINGW64__) && defined(__aarch64__))
    // global id = sum of the active processor counts of the lower groups + bit, the numbering ggml_win_cpu_to_group inverts
    const WORD n_groups = GetActiveProcessorGroupCount();
    std::vector<int32_t> group_base(n_groups + 1, 0);
    for (WORD g = 0; g < n_groups; ++g) {
        group_base[g + 1] = group_base[g] + (int32_t) GetActiveProcessorCount(g);
    }
    DWORD len = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len) && GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return cores;
    }
    std::vector<char> buf(len);
    if (len == 0 || !GetLogicalProcessorInformationEx(RelationProcessorCore,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
        return cores;
    }
    for (DWORD off = 0; off < len; ) {
        auto * info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
        if (info->Size == 0) {
            break;
        }
        if (info->Relationship == RelationProcessorCore) {
            std::vector<int32_t> core;
            for (WORD k = 0; k < info->Processor.GroupCount; ++k) {
                const GROUP_AFFINITY & ga = info->Processor.GroupMask[k];
                if (ga.Group >= n_groups) {
                    continue;
                }
                for (int bit = 0; bit < (int) (8 * sizeof(KAFFINITY)); ++bit) {
                    if (ga.Mask & ((KAFFINITY) 1 << bit)) {
                        core.push_back(group_base[ga.Group] + bit);
                    }
                }
            }
            if (!core.empty()) {
                cores.push_back(std::move(core));
            }
        }
        off += info->Size;
    }
#elif defined(__linux__)
    const long n_conf = sysconf(_SC_NPROCESSORS_CONF);
    const int  n_cpu  = (int) std::min<long>(n_conf > 0 ? n_conf : GGML_MAX_N_THREADS, GGML_MAX_N_THREADS);
    std::vector<std::string> keys; // sibling-list text of each core, in first-seen order
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        std::string key;
        if (!f.is_open() || !std::getline(f, key) || key.empty()) {
            continue; // offline cpu or hole in the numbering
        }
        size_t i = 0;
        while (i < keys.size() && keys[i] != key) {
            ++i;
        }
        if (i == keys.size()) {
            keys.push_back(key);
            cores.emplace_back();
        }
        cores[i].push_back(cpu);
    }
#endif
    return cores;
}

std::vector<int32_t> common_cpu_first_cpu_per_core(const std::vector<std::vector<int32_t>> & cores) {
    std::vector<int32_t> cpus;
    for (const auto & core : cores) {
        if (!core.empty()) {
            cpus.push_back(*std::min_element(core.begin(), core.end()));
        }
    }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

bool common_cpu_placement_parse(const std::string & value, bool & physical) {
    if (value == "physical") { physical = true;  return true; }
    if (value == "none")     { physical = false; return true; }
    return false;
}

bool common_cpu_apply_physical_placement(common_cpu_params & cpuparams, const std::vector<int32_t> & cpus) {
    if (!cpuparams.place_physical || cpuparams.mask_valid || cpus.empty()) {
        return false;
    }
    bool any = false;
    for (int32_t cpu : cpus) {
        if (cpu >= 0 && cpu < GGML_MAX_N_THREADS) {
            cpuparams.cpumask[cpu] = true;
            any = true;
        }
    }
    if (any) {
        cpuparams.mask_valid = true;
        cpuparams.strict_cpu = true;
    }
    return any;
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return !!(edx & (1u << 15));
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

struct cpu_affinity_restore {
    cpu_set_t prev;
    bool      active = false;

    ~cpu_affinity_restore() {
        if (active) {
            pthread_setaffinity_np(pthread_self(), sizeof(prev), &prev);
        }
    }
};

// cores of one class (P/E): primaries, then SMT siblings
static std::vector<int32_t> cpu_detect_cores(bool with_siblings, bool efficiency) {
    std::vector<int32_t> primaries;
    std::vector<int32_t> extra;

    int n_cpu = sysconf(_SC_NPROCESSORS_CONF); // covers holes from offline cpus
    if (n_cpu < 1 || !is_hybrid_cpu()) {
        return primaries;
    }
    n_cpu = std::min(n_cpu, (int) CPU_SETSIZE);

    cpu_affinity_restore restore;
    if (pthread_getaffinity_np(pthread_self(), sizeof(restore.prev), &restore.prev)) {
        return primaries;
    }
    restore.active = true;

    const cpu_set_t & affinity = restore.prev;

    std::unordered_set<std::string> seen;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (!CPU_ISSET(cpu, &affinity)) { // stay within the affinity we were started with
            continue;
        }
        if (pin_cpu(cpu)) {
            primaries.clear();
            extra.clear();
            break;
        }
        if (is_running_on_efficiency_core() != efficiency) {
            // E-cores harm lockstep threading
            continue;
        }

        const std::string key = cpu_physical_core_key(cpu);
        if (key.empty()) {
            continue; // cannot tell siblings apart, better skip than double count
        }
        if (seen.insert(key).second) {
            primaries.push_back(cpu);
        } else if (with_siblings) {
            extra.push_back(cpu);
        }
    }

    primaries.insert(primaries.end(), extra.begin(), extra.end());
    return primaries;
}

std::vector<int32_t> cpu_get_math_cpus() {
    static const std::vector<int32_t> cpus = cpu_detect_cores(false, false);
    return cpus;
}

std::vector<int32_t> cpu_get_efficiency_cpus() {
    static const std::vector<int32_t> cpus = cpu_detect_cores(false, true);
    return cpus;
}

static std::vector<int32_t> cpu_affinity_auto_cpus() {
    static const std::vector<int32_t> cpus = cpu_detect_cores(true, false);
    return cpus;
}

static std::vector<int32_t> cpu_affinity_auto_cpus_draft() {
    static const std::vector<int32_t> cpus = cpu_detect_cores(true, true);
    return cpus;
}

// drop CPUs outside the process affinity (taskset/cgroup)
static std::vector<int32_t> cpu_affinity_filter(const std::vector<int32_t> & cpus) {
    if (cpus.empty()) {
        return cpus;
    }

    cpu_set_t affinity;
    if (pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
        return cpus;
    }

    std::vector<int32_t> result;
    for (const int32_t cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &affinity)) {
            result.push_back(cpu);
        } else {
            fprintf(stderr, "warning: CPU %d is outside the process affinity, ignored\n", cpu);
        }
    }
    return result;
}

#else
std::vector<int32_t> cpu_get_math_cpus()       { return {}; }
std::vector<int32_t> cpu_get_efficiency_cpus() { return {}; }
static std::vector<int32_t> cpu_affinity_auto_cpus()       { return {}; }
static std::vector<int32_t> cpu_affinity_auto_cpus_draft() { return {}; }
static std::vector<int32_t> cpu_affinity_filter(const std::vector<int32_t> & cpus) { return cpus; }
#endif // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t common_cpu_get_num_math() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    const std::vector<int32_t> cpus = cpu_get_math_cpus();
    if (!cpus.empty()) {
        return (int32_t) cpus.size();
    }
#elif defined(__powerpc64__) || defined(__powerpc__)
    int32_t smt_factor = 1;
    int phy_cpus = common_cpu_get_num_physical_cores();
    int logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (phy_cpus > 0 && logical_cpus > phy_cpus) {
        smt_factor = logical_cpus / phy_cpus;
    }
    return phy_cpus * std::min(smt_factor, 2);
#endif
    return common_cpu_get_num_physical_cores();
}

// --- FORK-ONLY, carried verbatim from ours in this merge (declared in the already-
//     resolved common.h; upstream/theirs has no equivalent) ---

static std::vector<int32_t> cpu_affinity_resolve_impl(
        const std::vector<int32_t> & cpus, bool auto_detect, const std::vector<int32_t> & auto_cpus) {
    const std::vector<int32_t> resolved = (!cpus.empty() || !auto_detect) ? cpus : auto_cpus;
    return cpu_affinity_filter(resolved);
}

std::vector<int32_t> cpu_affinity_resolve(const std::vector<int32_t> & cpus, bool auto_detect) {
    return cpu_affinity_resolve_impl(cpus, auto_detect, cpu_affinity_auto_cpus());
}

std::vector<int32_t> cpu_affinity_resolve_draft(const std::vector<int32_t> & cpus, bool auto_detect) {
    return cpu_affinity_resolve_impl(cpus, auto_detect, cpu_affinity_auto_cpus_draft());
}

// Parse a CPU bitmask ("0x55", "85") into a list of logical CPU ids.
bool cpu_affinity_parse_mask(const std::string & value, std::vector<int32_t> & cpus) {
    if (value.find('-') != std::string::npos) {
        return false; // strtoull would wrap a negative value to ULLONG_MAX
    }

    errno = 0;
    char * end = nullptr;
    const unsigned long long mask = strtoull(value.c_str(), &end, 0);
    if (errno == ERANGE || end == value.c_str() || *end != '\0') {
        return false; // malformed or does not fit 64 CPUs
    }

    cpus.clear();
    for (int i = 0; i < 64; ++i) {
        if (mask & (1ull << i)) {
            cpus.push_back(i);
        }
    }
    return !cpus.empty();
}

// parse a non-negative decimal CPU id, without throwing
static bool cpu_affinity_parse_id(const std::string & str, int & cpu) {
    if (str.empty()) {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const long value = strtol(str.c_str(), &end, 10);
    if (errno != 0 || end == str.c_str() || *end != '\0' || value < 0 || value > INT_MAX) {
        return false;
    }

    cpu = (int) value;
    return true;
}

// Parse a CPU list ("0-3,8,10-11") into a list of logical CPU ids.
bool cpu_affinity_parse_range(const std::string & value, std::vector<int32_t> & cpus) {
    // bound to what the backend supports (also avoids absurd ranges)
    const int max_cpu = GGML_MAX_CPU_AFFINITY;

    std::vector<int32_t> result;
    std::stringstream ss(value);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            return false;
        }

        const size_t dash = item.find('-');
        if (dash == std::string::npos) {
            int cpu;
            if (!cpu_affinity_parse_id(item, cpu) || cpu >= max_cpu) {
                return false;
            }
            result.push_back(cpu);
        } else {
            int lo, hi;
            if (!cpu_affinity_parse_id(item.substr(0, dash), lo) ||
                !cpu_affinity_parse_id(item.substr(dash + 1), hi) ||
                hi < lo || hi >= max_cpu) {
                return false;
            }
            for (int cpu = lo; cpu <= hi; ++cpu) {
                result.push_back(cpu);
            }
        }
    }

    if (result.empty()) {
        return false;
    }

    cpus = std::move(result);
    return true;
}

//
// Arg utils
//
common_webui common_webui_from_name(const std::string& format) {
    if (format == "none") {
        return COMMON_WEBUI_NONE;
    }
    else if (format == "auto") {
        return COMMON_WEBUI_AUTO;
    }
    else if (format == "llamacpp") {
        return COMMON_WEBUI_LLAMACPP;
    }
    else {
        return COMMON_WEBUI_AUTO;
    }
}

common_checkpoint_eviction common_checkpoint_eviction_from_name(const std::string & format) {
    if (format == "auto") {
        return COMMON_CHECKPOINT_EVICTION_AUTO;
    } else if (format == "fifo") {
        return COMMON_CHECKPOINT_EVICTION_FIFO;
    } else if (format == "variance") {
        return COMMON_CHECKPOINT_EVICTION_VARIANCE;
    } else {
        return COMMON_CHECKPOINT_EVICTION_AUTO;
    }
}

thinking_tokens thinking_tokens_from_string(const std::string& format) {
    thinking_tokens think_token;
    std::string token_string = string_strip(format);
    if (token_string == "none" || token_string == "None") {
        think_token.exclude = false;
        return think_token;
    }
    else if (token_string == "auto" || token_string == "Auto") {
        think_token.exclude = true;
        think_token.begin = "<think>";
        think_token.end = "</think>";
        return think_token;
    }
    // Use user provided think tokens
    auto start_end = string_split(format, ",");
    if (start_end.size() == 2) {
        think_token.exclude = true;
        think_token.begin = start_end[0];
        think_token.end = start_end[1];
    }
    return think_token;
}



std::pair<int, char**> parse_command_line(const std::string& commandLine) {
    std::vector<std::string> tokens;
    std::string current;
    bool inQuotes = false;

    for (size_t i = 0; i < commandLine.length(); i++) {
        char c = commandLine[i];

        if (c == '\"') {
            inQuotes = !inQuotes;
        }
        else if (c == ' ' && !inQuotes) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        }
        else {
            current += c;
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    int argc = static_cast<int>(tokens.size());
    char** argv = new char* [static_cast<size_t>(argc) + 1];

    for (int i = 0; i < argc; i++) {
        argv[i] = new char[tokens[i].length() + 1];
        std::strcpy(argv[i], tokens[i].c_str());
    }
    argv[argc] = nullptr;
    return { argc, argv };
}

void free_command_line(int argc, char** argv) {
    if (argv == nullptr) return;

    for (int i = 0; i < argc; i++) {
        delete[] argv[i];
    }
    delete[] argv;
}



void gpt_params_handle_model_default(common_params & params) {
    if (!params.model.hf_repo.empty()) {
        // short-hand to avoid specifying --hf-file -> default it to --model
        if (params.model.hf_file.empty()) {
            if (params.model.empty()) {
                throw std::invalid_argument("error: --hf-repo requires either --hf-file or --model\n");
            }
            params.model.hf_file = params.model.path;
        } else if (params.model.empty()) {
            params.model.path = fs_get_cache_file(string_split(params.model.hf_file, "/").back()).string();
        }
    } else if (!params.model.url.empty()) {
        if (params.model.empty()) {
            auto f = string_split(params.model.url, "#").front();
            f = string_split(f, "?").front();
            params.model.path = fs_get_cache_file(string_split(f, "/").back()).string();
        }
    } else if (params.model.empty()) {
        params.model.path = DEFAULT_MODEL_PATH;
    }
}

//
// Environment variable utils
//

template<typename T>
static typename std::enable_if<std::is_same<T, std::string>::value, void>::type
get_env(std::string name, T & target) {
    char * value = std::getenv(name.c_str());
    target = value ? std::string(value) : target;
}

template<typename T>
static typename std::enable_if<!std::is_same<T, bool>::value && std::is_integral<T>::value, void>::type
get_env(std::string name, T & target) {
    char * value = std::getenv(name.c_str());
    target = value ? std::stoi(value) : target;
}

template<typename T>
static typename std::enable_if<std::is_floating_point<T>::value, void>::type
get_env(std::string name, T & target) {
    char * value = std::getenv(name.c_str());
    target = value ? std::stof(value) : target;
}

template<typename T>
static typename std::enable_if<std::is_same<T, bool>::value, void>::type
get_env(std::string name, T & target) {
    char * value = std::getenv(name.c_str());
    if (value) {
        std::string val(value);
        target = val == "1" || val == "true";
    }
}
void gpt_params_parse_from_env(common_params & params) {
    // we only care about server-related params for now
    get_env("LLAMA_ARG_MODEL",            params.model.path);
    get_env("LLAMA_ARG_MODEL_URL",        params.model.url);
    // ADAPTED 2026-09-13 alongside model_alias becoming std::set<std::string> (common.h). The
    // env var keeps the same comma-separated spelling the -a/--alias flag takes (arg.cpp:3091),
    // so `LLAMA_ARG_MODEL_ALIAS=a,b` and `-a a,b` now mean the same thing; a single name still
    // works and simply lands as a one-element set.
    {
        std::string alias_env;
        get_env("LLAMA_ARG_MODEL_ALIAS", alias_env);
        if (!alias_env.empty()) {
            for (auto & alias : string_split<std::string>(alias_env, ',')) {
                alias = string_strip(alias);
                if (!alias.empty()) {
                    params.model_alias.insert(alias);
                }
            }
        }
    }
    get_env("LLAMA_ARG_HF_REPO",          params.model.hf_repo);
    get_env("LLAMA_ARG_HF_FILE",          params.model.hf_file);
    get_env("LLAMA_ARG_THREADS",          params.n_threads);
    get_env("LLAMA_ARG_CTX_SIZE",         params.n_ctx);
    get_env("LLAMA_ARG_N_PARALLEL",       params.n_parallel);
    get_env("LLAMA_ARG_BATCH",            params.n_batch);
    get_env("LLAMA_ARG_UBATCH",           params.n_ubatch);
    get_env("LLAMA_ARG_N_GPU_LAYERS",     params.n_gpu_layers);
    get_env("LLAMA_ARG_THREADS_HTTP",     params.n_threads_http);
    get_env("LLAMA_ARG_CHAT_TEMPLATE",    params.chat_template);
    get_env("LLAMA_ARG_N_PREDICT",        params.n_predict);
    get_env("LLAMA_ARG_ENDPOINT_METRICS", params.endpoint_metrics);
    get_env("LLAMA_ARG_ENDPOINT_MEMORY",  params.endpoint_memory);
    get_env("LLAMA_ARG_ENDPOINT_SLOTS",   params.endpoint_slots);
    get_env("LLAMA_ARG_EMBEDDINGS",       params.embedding);
    get_env("LLAMA_ARG_FLASH_ATTN",       params.flash_attn);
    get_env("LLAMA_ARG_DEFRAG_THOLD",     params.defrag_thold);
    get_env("LLAMA_ARG_CONT_BATCHING",    params.cont_batching);
    {   // the fork keeps a list of listen hosts (hostnames); LLAMA_ARG_HOST names one
        std::string host;
        get_env("LLAMA_ARG_HOST", host);
        if (!host.empty()) {
            params.hostnames = { host };
        }
    }
    get_env("LLAMA_ARG_PORT",             params.port);
    get_env("LLAMA_ARG_CACHE_TYPE_K", params.cache_type_k);
    get_env("LLAMA_ARG_CACHE_TYPE_V", params.cache_type_v);
    // ADAPTED 2026-09-13. There is no `params.use_mlock` any more -- see the note at the
    // mparams.load_mode assignment below. The env var is KEPT rather than dropped: this
    // fork-local env path predates arg.cpp's own .set_env("LLAMA_ARG_MLOCK"), and silently
    // removing a documented env var is a behaviour change, not a build fix.
    //
    // The mapping is taken from arg.cpp:2727-2733, the --mlock handler in THIS tree, so the
    // env var and the flag cannot disagree: --mlock sets LLAMA_LOAD_MODE_MLOCK flat, NOT
    // MMAP_MLOCK. A bare `true` here therefore means mlock-without-mmap, exactly as the flag does.
    {
        bool mlock_env = false;
        get_env("LLAMA_ARG_MLOCK", mlock_env);
        if (mlock_env) {
            params.load_mode = LLAMA_LOAD_MODE_MLOCK;
        }
    }
    get_env("LLAMA_ARG_K_CACHE_HADAMARD", params.k_cache_hadamard);
    get_env("LLAMA_ARG_V_CACHE_HADAMARD", params.v_cache_hadamard);

}


// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    DWORD p = NORMAL_PRIORITY_CLASS;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = BELOW_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_NORMAL:   p = NORMAL_PRIORITY_CLASS;       break;
        case GGML_SCHED_PRIO_MEDIUM:   p = ABOVE_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_HIGH:     p = HIGH_PRIORITY_CLASS;         break;
        case GGML_SCHED_PRIO_REALTIME: p = REALTIME_PRIORITY_CLASS;     break;
    }

    if (!SetPriorityClass(GetCurrentProcess(), p)) {
        COM_WRN("failed to set process priority class %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#else // MacOS and POSIX
#include <sys/types.h>
#include <sys/resource.h>

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p =  5;  break;
        case GGML_SCHED_PRIO_NORMAL:   p =  0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = -5;  break;
        case GGML_SCHED_PRIO_HIGH:     p = -10; break;
        case GGML_SCHED_PRIO_REALTIME: p = -20; break;
    }

    if (setpriority(PRIO_PROCESS, 0, p) != 0) {
        COM_WRN("failed to set process priority %d : %s (%d)\n", prio, strerror(errno), errno);
        return false;
    }
    return true;
}

#endif

//
// CLI argument parsing
//


void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model) {
    int32_t n_set = 0;
    bool inherited = false;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model != nullptr) {
            inherited = true;
            // a role that only set --cpu-placement(-draft) keeps it when it inherits the rest from its role model
            const bool own_physical = cpuparams.place_physical;
            cpuparams = *role_model;
            if (own_physical) {
                cpuparams.place_physical = true;
            }
        } else {
            cpuparams.n_threads = common_cpu_get_num_math();
        }
    }

    // --cpu-placement physical: an explicit mask for this role (-C/-Cr, -Cb/-Crb, -Cd/-Crd ...) was parsed into
    // mask_valid already and wins; otherwise pin worker i to the i-th physical core (strict).
    if (cpuparams.place_physical) {
        if (cpuparams.mask_valid) {
            if (!inherited) { // an inherited mask was already reported for its role model
                COM_INF("%s", "an explicit CPU mask is set; --cpu-placement physical left it as given\n");
            }
        } else {
            const std::vector<int32_t> cpus = common_cpu_first_cpu_per_core(common_cpu_core_siblings());
            if (common_cpu_apply_physical_placement(cpuparams, cpus)) {
                COM_INF("--cpu-placement physical: %zu physical cores, strict placement, one worker per core\n", cpus.size());
            } else {
                COM_WRN("%s", "--cpu-placement physical: the core topology could not be read on this OS, placement left as default\n");
            }
        }
    }

    for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (cpuparams.cpumask[i]) {
            n_set++;
        }
    }

    if (n_set && n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        COM_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", n_set, cpuparams.n_threads);
    }
}

bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        COM_ERR("%s", "Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
        return false;
    }

    size_t start_i;
    size_t end_i;

    if (dash_loc == 0) {
        start_i = 0;
    } else {
        start_i = std::stoull(range.substr(0, dash_loc));
        if (start_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "Start index out of bounds!\n");
            return false;
        }
    }

    if (dash_loc == range.length() - 1) {
        end_i = GGML_MAX_N_THREADS - 1;
    } else {
        end_i = std::stoull(range.substr(dash_loc + 1));
        if (end_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "End index out of bounds!\n");
            return false;
        }
    }

    for (size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    // A hex digit carries 4 bits and boolmask has GGML_MAX_N_THREADS of them, so only the GGML_MAX_N_THREADS / 4 least significant
    // digits can name a CPU. Read from the least significant end: digit k (from the right) holds bits 4k .. 4k+3. A digit past that
    // limit is a leading zero (harmless) or names a CPU the thread array cannot hold (refused, never dropped).
    constexpr size_t max_digits = GGML_MAX_N_THREADS / 4;
    const size_t num_digits = mask.length() - start_i;

    for (size_t k = 0; k < num_digits; k++) {
        const size_t i = mask.length() - 1 - k;
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            COM_ERR("Invalid hex character '%c' at position %d\n", c, int32_t(i));
            return false;
        }

        if (k >= max_digits) {
            if (id != 0) {
                COM_ERR("CPU mask has a bit set beyond CPU %d (GGML_MAX_N_THREADS), at position %d\n", int32_t(GGML_MAX_N_THREADS - 1), int32_t(i));
                return false;
            }
            continue;
        }

        const size_t n = k * 4 + 3;
        boolmask[  n  ] = boolmask[  n  ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

void common_init() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);

    llama_log_set(common_log_default_callback, NULL);
}

void common_params_print_info(const common_params & params, bool print_devices) {
#ifdef NDEBUG
    const char * build_type = "";
#else
    const char * build_type = " (debug)";
#endif
    COM_TRC("%s: build %d (%s) with %s for %s%s\n", __func__, llama_build_number(), llama_commit(), llama_compiler(), llama_build_target(), build_type);

    const int verbosity = common_log_get_verbosity_thold();
    COM_INF("%s: verbosity = %d (adjust with the `-lv N` CLI arg)\n", __func__, verbosity);
    COM_INF("%s: log format = %s (adjust with the `--log-format` CLI arg)\n", __func__, params.log_json ? "json" : "text");

    // device enumeration creates a primary context on CUDA backends, skip it when the caller does not own any device
    if (print_devices && verbosity >= LOG_LEVEL_TRACE) {
        COM_TRC("%s", "device_info:\n");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            size_t free, total;
            ggml_backend_dev_memory(dev, &free, &total);
            COM_TRC("  - %-8s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
        }
    }
    COM_TRC("%s\n", common_params_get_system_info(params).c_str());
}

std::string common_params_get_system_info(const common_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.cpuparams.n_threads;
    if (params.cpuparams_batch.n_threads != -1) {
        os << " (n_threads_batch = " << params.cpuparams_batch.n_threads << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !(defined(__MINGW64__) && defined(__aarch64__)) // windows 7 and later
    // x86_64 MinGW-w64 declares GetActiveProcessorCount; only arm64 MinGW keeps upstream's exclusion
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " | " << llama_print_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
#endif

    return os.str();
}

//
// String utils
//

std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string string_strip(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    return str.substr(start, end - start);
}

std::string string_lcs(std::string_view a, std::string_view b) {
    if (a.empty() || b.empty()) return {};

    std::vector<std::vector<size_t>> dp(a.size() + 1, std::vector<size_t>(b.size() + 1, 0));
    size_t best_len = 0;
    size_t best_end_a = 0;

    for (size_t i = 1; i <= a.size(); ++i) {
        for (size_t j = 1; j <= b.size(); ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if (dp[i][j] > best_len) {
                    best_len = dp[i][j];
                    best_end_a = i;
                }
            }
        }
    }
    return std::string(a.substr(best_end_a - best_len, best_len));
}

std::string string_get_sortable_timestamp() {
    using clock = std::chrono::system_clock;

    const clock::time_point current_time = clock::now();
    const time_t as_time_t = clock::to_time_t(current_time);
    char timestamp_no_ns[100];
    std::strftime(timestamp_no_ns, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&as_time_t));

    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        current_time.time_since_epoch() % 1000000000).count();
    char timestamp_ns[11];
    snprintf(timestamp_ns, 11, "%09" PRId64, ns);

    return std::string(timestamp_no_ns) + "." + std::string(timestamp_ns);
}

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string regex_escape(const std::string & s) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(s, special_chars, "\\$&");
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

std::string string_from(bool value) {
    return value ? "true" : "false";
}

std::string string_from(const std::vector<int> & values) {
    std::stringstream buf;

    buf << "[ ";
    bool first = true;
    for (auto e : values) {
        if (first) {
            first = false;
        } else {
            buf << ", ";
        }
        buf << std::to_string(e);
    }
    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, token);

        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }

    buf << " ]";

    return buf.str();
}

void string_process_escapes(std::string & input) {
    std::size_t input_len = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':  input[output_idx++] = '\n'; break;
                case 'r':  input[output_idx++] = '\r'; break;
                case 't':  input[output_idx++] = '\t'; break;
                case '\'': input[output_idx++] = '\''; break;
                case '\"': input[output_idx++] = '\"'; break;
                case '\\': input[output_idx++] = '\\'; break;
                case 'x':
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const char x[3] = { input[input_idx + 1], input[input_idx + 2], 0 };
                        char *err_p = nullptr;
                        const long val = std::strtol(x, &err_p, 16);
                        if (err_p == x + 2) {
                            input_idx += 2;
                            input[output_idx++] = char(val);
                            break;
                        }
                    }
                    // fall through
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr || sep - data >= 128) {
        COM_ERR("%s: malformed KV override '%s'\n", __func__, data);
        return false;
    }
    llama_model_kv_override kvo;
    std::strncpy(kvo.key, data, sep - data);
    kvo.key[sep - data] = 0;
    sep++;
    if (strncmp(sep, "int:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.val_i64 = std::atol(sep);
    } else if (strncmp(sep, "float:", 6) == 0) {
        sep += 6;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.val_f64 = std::atof(sep);
    } else if (strncmp(sep, "bool:", 5) == 0) {
        sep += 5;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (std::strcmp(sep, "true") == 0) {
            kvo.val_bool = true;
        } else if (std::strcmp(sep, "false") == 0) {
            kvo.val_bool = false;
        } else {
            COM_ERR("%s: invalid boolean value for KV override '%s'\n", __func__, data);
            return false;
        }
    } else if (strncmp(sep, "str:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (strlen(sep) > 127) {
            COM_ERR("%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
            return false;
        }
        strncpy(kvo.val_str, sep, 127);
        kvo.val_str[127] = '\0';
    } else {
        COM_ERR("%s: invalid type for KV override '%s'\n", __func__, data);
        return false;
    }
    overrides.emplace_back(std::move(kvo));
    return true;
}

// Go time.ParseDuration grammar, as ollama parses keep_alive: [-+]?(<number><unit>)+ or "0",
// units ns us µs μs ms s m h. A negative duration means "forever" and yields -1.
bool string_parse_duration_ms(const std::string & str, int64_t & out_ms, bool bare_number_is_seconds) {
    std::string s = string_strip(str);
    if (s.empty()) {
        return false;
    }
    if (bare_number_is_seconds) {
        // OLLAMA_KEEP_ALIVE falls back to integer seconds when it is not a duration
        char * end = nullptr;
        errno = 0;
        const long long v = std::strtoll(s.c_str(), &end, 10);
        if (errno == 0 && end != s.c_str() && *end == '\0') {
            out_ms = v < 0 ? -1 : (int64_t) v * 1000;
            return true;
        }
    }
    size_t i = 0;
    bool neg = false;
    if (s[i] == '-' || s[i] == '+') {
        neg = s[i] == '-';
        i++;
    }
    if (s.substr(i) == "0") {
        out_ms = 0;
        return true;
    }
    if (i >= s.size()) {
        return false;
    }
    double total_ns = 0.0;
    while (i < s.size()) {
        const size_t num_start = i;
        while (i < s.size() && (std::isdigit((unsigned char) s[i]) || s[i] == '.')) {
            i++;
        }
        if (i == num_start) {
            return false; // a unit with no number
        }
        const std::string num = s.substr(num_start, i - num_start);
        if (num == "." || std::count(num.begin(), num.end(), '.') > 1) {
            return false;
        }
        const size_t unit_start = i;
        while (i < s.size() && !std::isdigit((unsigned char) s[i]) && s[i] != '.') {
            i++;
        }
        const std::string unit = s.substr(unit_start, i - unit_start);
        double scale;
        if      (unit == "ns")                                      { scale = 1.0; }
        else if (unit == "us" || unit == "\xC2\xB5s" || unit == "\xCE\xBCs") { scale = 1e3; }
        else if (unit == "ms")                                      { scale = 1e6; }
        else if (unit == "s")                                       { scale = 1e9; }
        else if (unit == "m")                                       { scale = 60e9; }
        else if (unit == "h")                                       { scale = 3600e9; }
        else { return false; } // missing or unknown unit
        total_ns += std::strtod(num.c_str(), nullptr) * scale;
    }
    if (neg && total_ns > 0.0) {
        out_ms = -1;
        return true;
    }
    out_ms = (int64_t) (total_ns / 1e6);
    return true;
}

static inline bool glob_class_match(const char c, const char * pattern, const char * class_end) {
    const char * class_start = pattern;
    bool negated = false;

    if (*class_start == '!') {
        negated = true;
        class_start++;
    }

    // If first character after negation is ']' or '-', treat it as literal
    if (*class_start == ']' || *class_start == '-') {
        if (class_start < class_end && *class_start == c) {
            return !negated;
        }
        class_start++;
    }

    bool matched = false;

    while (class_start < class_end) {
        if (class_start + 2 < class_end && class_start[1] == '-' && class_start[2] != ']') {
            char start_char = *class_start;
            char end_char = class_start[2];
            if (c >= start_char && c <= end_char) {
                matched = true;
                break;
            }
            class_start += 3;
        } else {
            if (*class_start == c) {
                matched = true;
                break;
            }
            class_start++;
        }
    }

    return negated ? !matched : matched;
}

// simple glob: * matches non-/ chars, ** matches anything including /, [] matches character class
static inline bool glob_match(const char * pattern, const char * str) {
    if (*pattern == '\0') {
        return *str == '\0';
    }
    if (pattern[0] == '*' && pattern[1] == '*') {
        const char * p = pattern + 2;
        if (glob_match(p, str)) return true;
        if (*str != '\0') return glob_match(pattern, str + 1);
        return false;
    }
    if (*pattern == '*') {
        const char * p = pattern + 1;
        for (; *str != '\0' && *str != '/'; str++) {
            if (glob_match(p, str)) return true;
        }
        return glob_match(p, str);
    }
    if (*pattern == '?' && *str != '\0' && *str != '/') {
        return glob_match(pattern + 1, str + 1);
    }
    if (*pattern == '[') {
        const char * class_end = pattern + 1;
        // If first character after '[' is ']' or '-', treat it as literal
        if (*class_end == ']' || *class_end == '-') {
            class_end++;
        }
        while (*class_end != '\0' && *class_end != ']') {
            class_end++;
        }
        if (*class_end == ']') {
            if (*str == '\0') return false;
            bool matched = glob_class_match(*str, pattern + 1, class_end);
            return matched && glob_match(class_end + 1, str + 1);
        } else {
            if (*str == '[') {
                return glob_match(pattern + 1, str + 1);
            }
            return false;
        }
    }
    if (*pattern == *str) {
        return glob_match(pattern + 1, str + 1);
    }
    return false;
}

bool glob_match(const std::string & pattern, const std::string & str) {
    return glob_match(pattern.c_str(), str.c_str());
}

void string_assign_append(std::string& dst, const std::string_view& sv, const std::string& str, const int32_t pos) {
    const int32_t margin = SSIZE(sv) - pos;
    if (margin > 0) {
        dst.assign(sv, pos).append(str);
    } else {
        dst.assign(str, -margin);
    }
}

//
// Filesystem utils
//

// Windows resolves a name whose stem (the part before the first '.', trailing spaces dropped) is a legacy DEVICE name to that device,
// whatever its extension or folder: saving "NUL" or "nul.bin" opens the null device instead of creating a file. Matched case-insensitively,
// including the superscript digits Windows also accepts for COM/LPT (UTF-8 C2 B9 / C2 B2 / C2 B3).
// SOURCE-READ: Strata serve/server.py:2101-2128 (MIT) and Microsoft "Naming Files, Paths, and Namespaces"; re-implemented here, no code copied.
static bool fs_is_windows_device_name(const std::string & component) {
    const size_t dot = component.find('.');
    std::string stem = component.substr(0, dot);
    while (!stem.empty() && stem.back() == ' ') {
        stem.pop_back();
    }
    for (char & ch : stem) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = (char) (ch - 'A' + 'a');
        }
    }
    if (stem == "con" || stem == "prn" || stem == "aux" || stem == "nul" || stem == "conin$" || stem == "conout$") {
        return true;
    }
    if (stem.size() >= 4 && (stem.compare(0, 3, "com") == 0 || stem.compare(0, 3, "lpt") == 0)) {
        const std::string tail = stem.substr(3);
        return (tail.size() == 1 && tail[0] >= '1' && tail[0] <= '9') ||
               tail == "\xC2\xB9" || tail == "\xC2\xB2" || tail == "\xC2\xB3";
    }
    return false;
}

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fs_validate_filename(const std::string & filename, bool allow_subdirs) {
    if (!filename.length()) {
        // Empty filename invalid
        return false;
    }
    if (filename.length() > 255) {
        // Limit at common largest possible filename on Linux filesystems
        // to avoid unnecessary further validation
        // (On systems with smaller limits it will be caught by the OS)
        return false;
    }

    size_t offset = 0;
    while (offset < filename.size()) {
        utf8_parse_result result = common_parse_utf8_codepoint(filename, offset);

        if (result.status != utf8_parse_result::SUCCESS) {
            return false;
        }
        uint32_t c = result.codepoint;

        if ((result.bytes_consumed == 2 && c < 0x80) ||
            (result.bytes_consumed == 3 && c < 0x800) ||
            (result.bytes_consumed == 4 && c < 0x10000)) {
            return false;
        }

        // Check for forbidden codepoints:
        // - Control characters
        // - Unicode equivalents of illegal characters
        // - UTF-16 surrogate pairs
        // - UTF-8 replacement character
        // - Byte order mark (BOM)
        // - Illegal characters: / \ : * ? " < > |
        if (c <= 0x1F // Control characters (C0)
            || c == 0x7F // Control characters (DEL)
            || (c >= 0x80 && c <= 0x9F) // Control characters (C1)
            || c == 0xFF0E // Fullwidth Full Stop (period equivalent)
            || c == 0x2215 // Division Slash (forward slash equivalent)
            || c == 0x2216 // Set Minus (backslash equivalent)
            || (c >= 0xD800 && c <= 0xDFFF) // UTF-16 surrogate pairs
            || c > 0x10FFFF // Max Unicode limit
            || c == 0xFFFD // Replacement Character (UTF-8)
            || c == 0xFEFF // Byte Order Mark (BOM)
            || c == ':' || c == '*' // Illegal characters
            || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            return false;
        }
        if (!allow_subdirs && (c == '/' || c == '\\')) {
            // Subdirectories not allowed, reject path separators
            return false;
        }
        offset += result.bytes_consumed;
    }

    // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
    // Unicode and other whitespace is not affected, only 0x20 space
    if (filename.front() == ' ' || filename.back() == ' ' || filename.back() == '.') {
        return false;
    }

    // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
    if (filename.find("..") != std::string::npos) {
        return false;
    }

    // Reject "."
    if (filename == ".") {
        return false;
    }

    // Reject a Windows device name (checked on every OS so a name saved here stays portable); each component of a sub-directory path counts
    size_t start = 0;
    while (start <= filename.size()) {
        size_t end = allow_subdirs ? filename.find_first_of("/\\", start) : std::string::npos;
        if (end == std::string::npos) {
            end = filename.size();
        }
        if (fs_is_windows_device_name(filename.substr(start, end - start))) {
            return false;
        }
        start = end + 1;
    }

    return true;
}

#include <iostream>


#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string & str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}
#endif // _WIN32 (utf8_to_wstring); fs_create_directory_with_parents below must exist on every platform

// returns true if successful, false otherwise
bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring wpath = utf8_to_wstring(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);

        pos_slash += 1;

        // skip the drive letter, in some systems it can return an access denied error
        if (subpath.length() == 2 && subpath[1] == ':') {
            continue;
        }

        const bool success = CreateDirectoryW(subpath.c_str(), NULL);

        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        } else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

#ifdef _WIN32
std::string wstring_to_utf8(const std::wstring & str) {
    if (str.empty()) {
        return std::string();
    }

    int size = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0, NULL, NULL);

    if (size <= 0) {
        return std::string();
    }

    std::string utf8(size, 0);
    WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), &utf8[0], size, NULL, NULL);

    return utf8;
}
#endif

void fs_write_atomic(const std::filesystem::path & path, const std::string & data) {
    // a random suffix gives every writer its own temporary file, across processes and containers
    std::random_device rd;
    const uint64_t id = ((uint64_t) rd() << 32) | rd();

    std::error_code ec;
    std::filesystem::path path_tmp = path;
    path_tmp += "." + std::to_string(id) + ".tmp";

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
    }

    std::ofstream file(path_tmp, std::ios::binary);
    file << data;
    file.close();

    if (!file.fail()) {
        std::filesystem::rename(path_tmp, path, ec);
    }

    if (file.fail() || ec) {
        std::filesystem::remove(path_tmp, ec);
        throw std::runtime_error("failed to write file: " + path.string());
    }
}

bool fs_is_directory(const std::string & path) {
    std::filesystem::path dir(path);
    return std::filesystem::exists(dir) && std::filesystem::is_directory(dir);
}

std::string common_get_env(const std::string & name) {
    const char * value = std::getenv(name.c_str());
    return value == nullptr ? "" : value;
}

void common_set_env(const std::string & name, const std::string & value) {
#if defined(_WIN32)
    _putenv_s(name.c_str(), value.c_str());
#else
    if (value.empty()) {
        unsetenv(name.c_str());
    } else {
        setenv(name.c_str(), value.c_str(), 1);
    }
#endif
}

#if !defined(_WIN32)
static std::filesystem::path get_home_directory() {
    std::filesystem::path home = common_get_env("HOME");
    if (!home.empty()) {
        return home;
    }
    const struct passwd * pw = getpwuid(getuid());
    if (!pw || !pw->pw_dir || !*pw->pw_dir) {
        throw std::runtime_error("Failed to find $HOME directory");
    }
    return pw->pw_dir;
}
#endif

std::filesystem::path fs_get_cache_directory() {
    std::filesystem::path cache_directory = common_get_env("LLAMA_CACHE");
    if (!cache_directory.empty()) {
        return cache_directory;
    }
#if defined(_WIN32)
    cache_directory = common_get_env("LOCALAPPDATA");
    if (cache_directory.empty()) {
        throw std::runtime_error("Failed to find %LOCALAPPDATA% directory");
    }
#elif defined(__APPLE__)
    cache_directory = get_home_directory() / "Library/Caches";
#else
    cache_directory = common_get_env("XDG_CACHE_HOME");
    if (cache_directory.empty()) {
        cache_directory = get_home_directory() / ".cache";
    }
#endif
    return cache_directory / "llama.cpp";
}

std::filesystem::path fs_get_config_directory() {
    std::filesystem::path config_directory;
#if defined(_WIN32)
    config_directory = common_get_env("APPDATA");
    if (config_directory.empty()) {
        throw std::runtime_error("Failed to find %APPDATA% directory");
    }
#else
    config_directory = common_get_env("XDG_CONFIG_HOME");
    if (config_directory.empty()) {
        config_directory = get_home_directory() / ".config";
    }
#endif
    return config_directory / "llama.cpp";
}

std::filesystem::path fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    const std::filesystem::path cache_directory = fs_get_cache_directory();
    std::error_code ec;
    common_create_directories(cache_directory, ec);
    if (ec) {
        throw std::runtime_error("failed to create cache directory: " + cache_directory.string());
    }
    return cache_directory / std::filesystem::u8path(filename);
}

//
// TTY utils
//

bool common_is_tty(FILE * file) {
#if defined(_WIN32)
    return _isatty(_fileno(file));
#else
    return isatty(fileno(file));
#endif
}

bool tty_can_use_colors() {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (const char * no_color = std::getenv("NO_COLOR")) {
        if (no_color[0] != '\0') {
            return false;
        }
    }

    // Check TERM environment variable
    if (const char * term = std::getenv("TERM")) {
        if (std::strcmp(term, "dumb") == 0) {
            return false;
        }
    }

    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    return common_is_tty(stdout) || common_is_tty(stderr);
}

bool tty_enable_ansi() {
#if defined(_WIN32)
    // a Windows console renders ANSI sequences only in virtual terminal mode, pipes and files take them as is
    for (DWORD id : { STD_OUTPUT_HANDLE, STD_ERROR_HANDLE }) {
        HANDLE h    = GetStdHandle(id);
        DWORD  mode = 0;
        if (GetConsoleMode(h, &mode) && !SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
            return false;
        }
    }
#endif
    return true;
}

//
// Model utils
//

// TODO: move to common/sampling
static void common_init_sampler_from_model(
    const llama_model * model,
    common_params_sampling & sparams) {

    const uint64_t config = sparams.user_sampling_config;

    auto get_int32 = [&](const char * key, int32_t & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[64] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            int32_t v = strtol(buf, &end, 10);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    auto get_float = [&](const char * key, float & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[128] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            float v = strtof(buf, &end);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    // Sampling sequence
    if (!(config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS)) {
        char buf[512] = {0};
        if (llama_model_meta_val_str(model, llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), buf, sizeof(buf)) > 0) {
            const std::vector<std::string> sampler_names = string_split<std::string>(std::string(buf), ';');
            if (!sampler_names.empty()) {
                // ADAPTED 2026-09-13 to the fork's own field and parser -- see the note at
                // arg.cpp's samplers_sequence loop. This line is byte-for-byte what this file
                // itself carried before the merge (common.cpp.pre-fmt-all:1486).
                sparams.samplers_sequence = llama_sampling_types_from_names(sampler_names, true);
            }
        }
    }

    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K),           sparams.top_k,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P),           sparams.top_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P),           sparams.min_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY), sparams.xtc_probability, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD),   sparams.xtc_threshold,   common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TEMP),            sparams.temp,            common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N),  sparams.penalty_last_n,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT),  sparams.penalty_repeat,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT),        sparams.mirostat,        common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU),    sparams.mirostat_tau,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA),    sparams.mirostat_eta,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA);
}

struct common_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top

    common_threadpools threadpools;

    llama_model_ptr   model;
    llama_context_ptr context;

    std::vector<llama_adapter_lora_ptr> lora;

    std::vector<common_sampler_ptr> samplers;
    std::vector<llama_sampler_seq_config> samplers_seq_config;
};

static const std::map<common_decision_type, std::string> COMMON_DECISION_TYPE_NAMES = {
    { COMMON_DECISION_TYPE_OPENJEV,        "openjev"       },
    { COMMON_DECISION_TYPE_LEV,            "lev"           },
    { COMMON_DECISION_TYPE_KEV,            "kev"           },
    { COMMON_DECISION_TYPE_NIMBLE,         "nimble"        },
    { COMMON_DECISION_TYPE_LAYA,           "laya"          },
    { COMMON_DECISION_TYPE_CLEF,           "clef"          },
    { COMMON_DECISION_TYPE_PPLX_DECIDER,   "pplx-decider"  },
    { COMMON_DECISION_TYPE_VLLM_DECISION2, "vllm-decision2" },
    { COMMON_DECISION_TYPE_VLLM_D3,        "vllm-d3"        },
    { COMMON_DECISION_TYPE_LFM2_D1,        "lfm2-d1"       },
    { COMMON_DECISION_TYPE_LFM2_D1_OMNI,   "lfm2-d1-omni"  },
    { COMMON_DECISION_TYPE_GRANITE_SWITCH, "granite-switch"},
    { COMMON_DECISION_TYPE_H2O_LIGHTNING,  "h2o-lightning" },
    // for future models, ALWAYS add org name prefix to avoid name collisions, e.g. "orgname-modelname"
};

static common_decision_type common_decision_type_from_string(const std::string & str) {
    for (const auto & pair : COMMON_DECISION_TYPE_NAMES) {
        if (pair.second == str) {
            return pair.first;
        }
    }
    return COMMON_DECISION_TYPE_UNKNOWN;
}

common_decision_type common_get_decision_type(const struct llama_model * model) {
    char buf[64];
    if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) < 0) {
        return COMMON_DECISION_TYPE_NONE;
    }
    const std::string key = std::string(buf) + ".decision.type";
    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) < 0) {
        return COMMON_DECISION_TYPE_NONE;
    }
    return common_decision_type_from_string(buf);
}

common_gguf_info common_get_gguf_info(const std::string & fname) {
    common_gguf_info info;

    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(fname.c_str(), gguf_params));
    if (!gguf_ctx) {
        return info; // missing or unreadable file
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return info; // no architecture in the metadata
    }
    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch.empty()) {
        return info;
    }

    const int64_t type_id = gguf_find_key(gguf_ctx.get(), (arch + ".decision.type").c_str());
    if (type_id < 0) {
        info.decision_type = COMMON_DECISION_TYPE_NONE;
    } else if (gguf_get_kv_type(gguf_ctx.get(), type_id) == GGUF_TYPE_STRING) {
        info.decision_type = common_decision_type_from_string(gguf_get_val_str(gguf_ctx.get(), type_id));
    }

    // same key and type as the model loader
    const int64_t ctx_id = gguf_find_key(gguf_ctx.get(), (arch + ".context_length").c_str());
    if (ctx_id >= 0 && gguf_get_kv_type(gguf_ctx.get(), ctx_id) == GGUF_TYPE_UINT32) {
        info.n_ctx_train = gguf_get_val_u32(gguf_ctx.get(), ctx_id);
    }

    // same key as llama_model_chat_template(model, nullptr)
    info.has_chat_tmpl = gguf_find_key(gguf_ctx.get(), "tokenizer.chat_template") >= 0;

    return info;
}

bool common_verify_model_checksum(const std::string & path, const std::string & expected_hex, std::string & err) {
    if (!common_is_sha256_hex(expected_hex)) {
        err = "expected value '" + expected_hex + "' is not a 64-character hex SHA-256";
        return false;
    }
    std::string want = expected_hex;
    for (char & c : want) {
        c = (char) std::tolower((unsigned char) c);
    }
    std::string got;
    if (!common_file_sha256_hex(path, got)) {
        err = "cannot read '" + path + "' to compute its SHA-256";
        return false;
    }
    if (got != want) {
        err = "SHA-256 mismatch for '" + path + "': expected " + want + ", actual " + got;
        return false;
    }
    return true;
}

size_t common_lora_adapters_tensor_bytes(const std::vector<common_adapter_lora_info> & adapters) {
    size_t total = 0;
    for (const auto & a : adapters) {
        struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * g = gguf_init_from_file(a.path.c_str(), gp);
        if (!g) {
            LOG_WRN("%s: cannot read lora adapter header '%s', not counted in the fit reserve\n", __func__, a.path.c_str());
            continue;
        }
        const int64_t n = gguf_get_n_tensors(g);
        for (int64_t i = 0; i < n; ++i) {
            total += gguf_get_tensor_size(g, i);
        }
        gguf_free(g);
    }
    return total;
}

size_t common_lora_fit_reserve_bytes(const std::vector<common_adapter_lora_info> & adapters, bool gpu_offload) {
    if (!gpu_offload || adapters.empty()) {
        return 0;
    }
    return common_lora_adapters_tensor_bytes(adapters);
}

common_init_result::common_init_result(common_params & params, bool model_only) :
    pimpl(new impl{}) {
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (params.fit_params) {
        COM_TRC("%s", "fitting params to device memory ...\n");
        COM_TRC("%s", "(for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)\n");

        // the draft context is created from the same base params and follows the main context, fit both together
        const bool has_draft = params.speculative.has_dft();
        // ADAPTED 2026-09-13. Upstream keeps a `types` vector and a fused DRAFT_MTP enumerator;
        // this fork keeps a STAGE LIST queried through common_params_speculative::has_stage_type
        // and keeps DRAFT and MTP as separate stage types (common.h:278,281).
        //
        // Not invented: speculative.cpp:1636 already computes this exact value, under this exact
        // name, for this exact purpose -- `const bool spec_mtp = params.speculative
        // .has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);` followed by `cparams.ctx_type =
        // LLAMA_CONTEXT_TYPE_MTP` three lines later. This is the same expression as its sibling.
        const bool spec_mtp  = params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);

        common_params params_dft = common_base_params_to_speculative(params);

        auto mparams_dft = common_model_params_to_llama(params_dft);
        auto cparams_dft = common_context_params_to_llama(params_dft);
        if (spec_mtp) {
            cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        }
        cparams_dft.n_rs_seq = 0;

        const common_fit_extra_model extra = {
            /*.path_model   =*/ params_dft.model.path.c_str(),
            /*.mparams      =*/ &mparams_dft,
            /*.cparams      =*/ &cparams_dft,
            /*.shares_model =*/ !has_draft, // an MTP context runs on the weights of the main model
        };

        // --moe-cache-mib allocates its device cache at context creation, and the fit's dry-run context does not create
        // it (llama_context::sched_reserve skips the cache under no_alloc), so the fit would hand that memory to layers or
        // context. Upstream ik (894191d6e common/common.cpp:924, src/llama.cpp:9222) rejects --fit + cache outright; here
        // fit is on by default, so the budget is reserved instead: added to every device's fit margin. That is exact on
        // one GPU and conservative on several (the cache splits its budget across GPUs, llama-moe-cache.cpp:256).
        std::vector<size_t> fit_target = params.fit_params_target;
        if (cparams.moe_cache_size > 0) {
            for (size_t & t : fit_target) {
                t += cparams.moe_cache_size;
            }
            LOG_INF("%s: --moe-cache-mib: reserving %.0f MiB per device in the fit margin\n",
                    __func__, cparams.moe_cache_size / (1024.0*1024.0));
        }

        // --lora adapters are loaded after the fit and stay resident even at scale 0, but the fit's dry run does not
        // see them: reserve their tensor bytes in every device's margin (conservative on several GPUs).
        const size_t lora_reserve = common_lora_fit_reserve_bytes(params.lora_adapters, mparams.n_gpu_layers != 0);
        if (lora_reserve > 0) {
            for (size_t & t : fit_target) {
                t += lora_reserve;
            }
            LOG_INF("%s: --lora: reserving %.0f MiB per device in the fit margin\n",
                    __func__, lora_reserve / (1024.0*1024.0));
        }

        const common_params_fit_status fit_status = common_fit_params(params.model.path.c_str(), &mparams, &cparams,
            params.tensor_split,
            params.tensor_buft_overrides.data(),
            fit_target.data(),
            params.fit_params_min_ctx,
            has_draft || spec_mtp ? &extra : nullptr,
            params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR,
            params.moe_cache_auto);

        // the fit writes its expert placement into params; keep the cache size it chose with it, so a reload from params
        // gets the same cache
        if (params.moe_cache_auto) {
            params.moe_cache_size = cparams.moe_cache_size;
            params.moe_cache_auto = false;
        }

        // Upstream discards this status (279b4bfaf^2:common/common.cpp:1320), so a fit that FAILED
        // still loads with whatever mparams hold. When -ngl was left on auto that is -1, i.e. EVERY
        // layer on the device -- the one choice fit has just said does not fit. On Windows the driver's
        // system-memory fallback lets that allocation succeed and the run crawls with the card at zero
        // headroom (2026-09-27, row S69-FIT-NGL-AUTO-OVERCOMMITS-VRAM-ON-HYBRID-27B-F16-2026-09-27-01).
        // So: an unresolved auto becomes CPU-only, loudly. An explicit -ngl is the user's decision and
        // is left alone.
        if (fit_status != COMMON_PARAMS_FIT_STATUS_SUCCESS && mparams.n_gpu_layers < 0) {
            LOG_WRN("%s: fitting params to device memory %s and -ngl is still 'auto' -- loading with "
                    "-ngl 0 (CPU only) instead of every layer on the device. Pass an explicit -ngl to override.\n",
                    __func__, fit_status == COMMON_PARAMS_FIT_STATUS_FAILURE ? "FAILED" : "hit an ERROR");
            mparams.n_gpu_layers = 0;
        }
    }

    if (!params.model_checksum.empty()) {
        std::string cks_err;
        if (!common_verify_model_checksum(params.model.path, params.model_checksum, cks_err)) {
            COM_ERR("--model-checksum: %s\n", cks_err.c_str());
            return;
        }
        COM_INF("--model-checksum: '%s' matches the expected SHA-256\n", params.model.path.c_str());
    }

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == NULL) {
        return;
    }

    // a predictor-only MTP GGUF has no main blocks, so it cannot be the target model (ik_llama #2328,
    // ported into this constructor: the fork's model load lives here, not in the old common_init_from_params body)
    if (llama_model_mtp_package(model) == LLAMA_MTP_PACKAGE_COMPANION) {
        COM_ERR("'%s' is an MTP companion, pass it with -md instead\n", params.model.path.c_str());
        llama_model_free(model);
        return;
    }

    pimpl->model.reset(model);

    if (model_only) {
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // these decision models return a score for each token via the embeddings output
    // TODO: maybe improve this in the future
    const auto decision_type = common_get_decision_type(model);
    if (decision_type == COMMON_DECISION_TYPE_LAYA || decision_type == COMMON_DECISION_TYPE_KEV || decision_type == COMMON_DECISION_TYPE_CLEF ||
        decision_type == COMMON_DECISION_TYPE_LFM2_D1_OMNI || decision_type == COMMON_DECISION_TYPE_VLLM_DECISION2) {
        params.embedding    = true;
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;

        cparams.embeddings            = true;
        cparams.pooling_type          = LLAMA_POOLING_TYPE_NONE;
        cparams.n_outputs_max         = cparams.n_batch;
        cparams.n_outputs_max_per_seq = 1;

        LOG_INF("%s", "decision model reads the embeddings output, enabling embedding mode\n");

        // --decision-budget auto: the encoder reads the whole prompt in one ubatch, so the ubatch is the real limit on the prompt
        // (default 512). Size it to the context, which is the training context of the model unless -c is given.
        if (decision_type == COMMON_DECISION_TYPE_LAYA && params.decision_budget != "model" && !params.n_ubatch_explicit) {
            const uint32_t n_fit = cparams.n_ctx > 0 ? cparams.n_ctx : (uint32_t) llama_model_n_ctx_train(model);
            if (n_fit > cparams.n_ubatch) {
                cparams.n_ubatch = n_fit;
                cparams.n_batch  = std::max(cparams.n_batch, n_fit);
                params.n_ubatch  = (int32_t) n_fit;
                params.n_batch   = (int32_t) cparams.n_batch;
                cparams.n_outputs_max = cparams.n_batch;
                LOG_INF("decision budget auto: n_ubatch = n_batch = %u (the context; training context of the model = %d)\n",
                        n_fit, (int) llama_model_n_ctx_train(model));
            }
        }
    }

    // embeddings need the whole batch in one ubatch, so n_batch must not be larger than n_ubatch
    // (server.cpp does this check for --embedding, but before the model is loaded)
    if (cparams.embeddings && cparams.n_batch > cparams.n_ubatch) {
        LOG_WRN("embeddings enabled: setting n_batch = n_ubatch = %u\n", cparams.n_ubatch);
        cparams.n_batch = cparams.n_ubatch;
        params.n_batch  = params.n_ubatch;
    }

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_adapter_lora_ptr lora;
        lora.reset(llama_adapter_lora_init(model, la.path.c_str()));
        if (lora == nullptr) {
            COM_ERR("failed to load lora adapter '%s'\n", la.path.c_str());
            return;
        }

        char buf[1024];
        la.ptr = lora.get();
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.task_name", buf, sizeof(buf));
        la.task_name = buf;
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.prompt_prefix", buf, sizeof(buf));
        la.prompt_prefix = buf;
        pimpl->lora.emplace_back(std::move(lora)); // copy to list of loaded adapters
    }

    // updates params.sampling
    // TODO: fix naming
    common_init_sampler_from_model(model, params.sampling);

    // --- FORK-ONLY, carried from ours: model-embedded sampling recommendations
    //     (general.sampling.*, upstream PR #17120). params.sparams -> params.sampling
    //     TRANSLATED (header renamed the field). Precedence enforced inside: an
    //     explicit user flag (tracked in args_seen) always wins.
    common_params_sampling_apply_model_kv(model, params.sampling, params.args_seen);

    // --sampling-family-defaults: vendor-recommended values per model family, only for a GGUF with no general.sampling.* key
    if (params.sampling_family_defaults) {
        std::vector<std::string> ids(params.model_alias.begin(), params.model_alias.end());
        for (const char * key : {"general.name", "general.basename"}) {
            char buf[256];
            const int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
            if (n > 0) {
                ids.emplace_back(buf, (size_t) std::min<int32_t>(n, (int32_t) sizeof(buf) - 1));
            }
        }
        ids.push_back(params.model.path);
        ids.push_back(params.model.hf_repo);
        ids.push_back(params.model.hf_file);
        common_params_sampling_apply_family_defaults(model, params.sampling, params.args_seen, ids);
    }

    if (params.sampling.ignore_eos && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        COM_WRN("%s", "vocab does not have an EOS token, ignoring --ignore-eos\n");
        params.sampling.ignore_eos = false;
    }

    // initialize once
    for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            COM_TRC("added %s logit bias = %f\n", common_token_to_piece(vocab, i).c_str(), -INFINITY);
            params.sampling.logit_bias_eog.push_back({i, -INFINITY});
        }
    }

    if (params.sampling.ignore_eos) {
        // add EOG biases to the active set of logit biases
        params.sampling.logit_bias.insert(
                params.sampling.logit_bias.end(),
                params.sampling.logit_bias_eog.begin(), params.sampling.logit_bias_eog.end());
    }

    // init the backend samplers as part of the context creation
    pimpl->samplers.resize(cparams.n_seq_max);
    pimpl->samplers_seq_config.resize(cparams.n_seq_max);

    for (int i = 0; i < (int) cparams.n_seq_max; ++i) {
        pimpl->samplers[i].reset(common_sampler_init(model, params.sampling));
        pimpl->samplers_seq_config[i] = { i, common_sampler_get(pimpl->samplers[i].get()) };
    }

    if (params.sampling.backend_sampling) {
        cparams.samplers   = pimpl->samplers_seq_config.data();
        cparams.n_samplers = pimpl->samplers_seq_config.size();
    }

    llama_context * lctx = llama_init_from_model(model, cparams);
    if (lctx == NULL) {
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return;
    }

    pimpl->context.reset(lctx);

    set_process_priority(params.cpuparams.priority);

    pimpl->threadpools.init(lctx, params);
}

llama_model * common_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_init_result::context() {
    return pimpl->context.get();
}

common_sampler * common_init_result::sampler(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= (int) pimpl->samplers.size()) {
        return nullptr;
    }
    return pimpl->samplers[seq_id].get();
}

void common_init_result::reset_samplers() {
    for (int i = 0; i < (int) pimpl->samplers.size(); ++i) {
        llama_sampler_reset(common_sampler_get(pimpl->samplers[i].get()));
    }
}

std::vector<llama_adapter_lora_ptr> & common_init_result::lora() {
    return pimpl->lora;
}

// --cram-probe: one always-on log line with what the read-only sysfs probe found (root nullptr = /sys). Returns the probe result.
llama_cram_probe_result common_cram_probe_log(const char * sysfs_root) {
    const llama_cram_probe_result r = llama_cram_probe(sysfs_root ? sysfs_root : "/sys");
    LOGV(LOG_LEVEL_OUTPUT, "%s: CRAM probe (--cram-probe): available=%d nodes=%d memory_only=%d private_or_compressed_signals=%d (%s) -- the sysfs names probed are GUESSES "
            "(the kernel series is not merged); a negative result does not show the hardware is absent\n", __func__,
            (int) r.available, r.node_count_total, r.node_count_memory_only, r.node_count_private_or_compressed, r.detail.c_str());
    return r;
}

bool common_vram_spill_apply(const common_params & params) {
    ggml_vram_spill_mode mode = GGML_VRAM_SPILL_OFF;
    if (!ggml_vram_spill_parse_mode(params.vram_spill_mode.c_str(), &mode)) {
        throw std::invalid_argument("--vram-spill: unknown mode '" + params.vram_spill_mode + "' (expected off|managed|hostmap|auto)");
    }
    if (mode == GGML_VRAM_SPILL_OFF) {
        return false; // default: touch nothing, log nothing
    }
    ggml_vram_spill_policy pol;
    pol.mode            = mode;
    pol.threshold_bytes = (uint64_t) std::max<int64_t>(params.vram_spill_threshold_mib, 0) * 1024ull * 1024ull;
    pol.reserve_bytes   = (uint64_t) std::max<int64_t>(params.vram_spill_reserve_mib,   0) * 1024ull * 1024ull;
    ggml_vram_spill_configure(&pol);
    // Runtime truth, not a compile-time guess: does any loaded backend publish the spill allocator?
    bool backend_can_spill = false;
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        if (ggml_backend_reg_get_proc_address(ggml_backend_reg_get(i), "ggml_backend_spill_buffer_type") != nullptr) {
            backend_can_spill = true;
        }
    }
    const char * effect = backend_can_spill
        ? "a loaded backend publishes the spill allocator: allocations >= threshold are routed by the policy (CUDA path UNTESTED on hardware)"
        : "NO loaded backend implements spill (CPU-only / non-CUDA build): the policy is recorded but nothing allocates through it, so it has no effect";
    LOGV(LOG_LEVEL_OUTPUT, "%s: vram-spill (--vram-spill): mode=%s threshold=%" PRId64 " MiB reserve=%" PRId64 " MiB -- %s\n", __func__,
            ggml_vram_spill_mode_name(mode), params.vram_spill_threshold_mib, params.vram_spill_reserve_mib, effect);
    return true;
}

// only for warmup and probe decodes, fill zeros as dummy input
static void common_batch_set_zero_state(common_batch & batch, const llama_model * model, std::vector<float> & zeros) {
    zeros.assign(llama_model_n_embd_out(model), 0.0f);
    for (int32_t i = 0; i < batch.size(); ++i) {
        batch.set_embd_state(i, { zeros.data(), 1, zeros.size() });
    }
}

common_init_result_ptr common_init_from_params(common_params & params, bool model_only) {
    common_vram_spill_apply(params);

    // Opt-in only: never touch the process-wide switch unless --cpu-flush-denormals was given.
    if (params.cpuparams.flush_denormals) {
        ggml_cpu_set_flush_denormals(true);
    }

    common_init_result_ptr res(new common_init_result(params, model_only));

    llama_model * model = res->model();
    if (model == NULL) {
        COM_ERR("failed to load model '%s'\n", params.model.path.c_str());
        return res;
    }

    if (params.cram_probe) {
        common_cram_probe_log(nullptr);
    }

    if (model_only) {
        return res;
    }

    llama_context * lctx = res->context();
    if (lctx == NULL) {
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return res;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (params.ctx_shift && !llama_memory_can_shift(llama_get_memory(lctx))) {
        COM_WRN("%s", "KV cache shifting is not supported for this context, disabling KV cache shifting\n");
        params.ctx_shift = false;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_model_n_layer(model);

        const auto cvec = common_control_vector_load(params.control_vectors, llama_model_n_layer(model));
        if (cvec.n_embd == -1) {
            return res;
        }

        int err = llama_set_adapter_cvec(
                lctx,
                cvec.data.data(),
                cvec.data.size(),
                cvec.n_embd,
                params.control_vector_layer_start,
                params.control_vector_layer_end);
        if (err) {
            return res;
        }
    }

    if (llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_RANK) {
        bool ok = true;

        if (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL) {
            COM_WRN("%s", "vocab does not have a  BOS token, reranking will not work\n");
            ok = false;
        }

        bool has_eos = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
        bool has_sep = llama_vocab_sep(vocab) != LLAMA_TOKEN_NULL;
        bool has_rerank_prompt = llama_model_chat_template(model, "rerank") != NULL;

        if (!has_eos && !has_sep && !has_rerank_prompt) {
            COM_WRN("%s", "vocab does not have an EOS token, SEP token, or rerank prompt. Reranking will not work\n");
            ok = false;
        } else if (!has_eos) {
            COM_WRN("%s", "vocab does not have an EOS token, using SEP token as fallback\n");
        }

        if (!ok) {
            return res;
        }
    }

    if (!params.lora_init_without_apply) {
        common_set_adapter_lora(lctx, params.lora_adapters);
    }

    if (params.warmup) {
        COM_TRC("%s", "warming up the model with an empty run - please wait ... (--no-warmup to disable)\n");

        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);

        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        } else if (eos != LLAMA_TOKEN_NULL) {
            // a model with neither BOS nor EOS must not get LLAMA_TOKEN_NULL here: common_batch reads an id of LLAMA_TOKEN_NULL as an
            // embedding row (get_sub_batch -> llama_batch_ext_add_embd with no data: abort, ctest test-speculative-rejection-sampling)
            tmp.push_back(eos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        // Upstream's 2-argument llama_batch_get_one (DEC-S69-FORK-ADOPT-UPSTREAM-BATCH-ABI-2026-09-26).
        // The context is fresh here and its memory is cleared right after, so the auto-tracked
        // position is 0 -- the batch the older 4-argument call built explicitly.
        if (llama_model_has_encoder(model)) {
            common_batch batch = common_batch_get_one(lctx, tmp);
            llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get());
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            tmp.resize(std::min(tmp.size(), (size_t) params.n_batch));
            common_batch batch = common_batch_get_one(lctx, tmp);
            std::vector<float> zeros;
            common_batch_set_zero_state(batch, model, zeros);
            llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        }
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);

        // measured per-device bytes next to what common_fit_params projected (no-op when no fit ran); WARN above +5 percent
        common_fit_log_measured(lctx);

        // reset samplers to reset RNG state after warmup to the seeded state
        res->reset_samplers();
    }

    return res;
}

common_init_result::~common_init_result() = default;

std::string common_get_model_endpoint() {
    std::string endpoint = common_get_env("MODEL_ENDPOINT");
    if (endpoint.empty()) {
        // the HF_ENDPOINT variable is respected for backward compatibility
        endpoint = common_get_env("HF_ENDPOINT");
    }
    if (endpoint.empty()) {
        return "https://huggingface.co/";
    }
    if (endpoint.back() != '/') {
        endpoint += '/';
    }
    return endpoint;
}

char * common_get_model_or_exit(int argc, char * argv[]) {
    if (argc > 1) {
        return argv[1];
    }

    char * path = getenv("LLAMACPP_TEST_MODELFILE");
    if (!path || strlen(path) == 0) {
        fprintf(stderr, "\033[33mWARNING: No model file provided. Skipping this test. Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> to silence this warning and run this test.\n\033[0m");
        exit(EXIT_SUCCESS);
    }

    return path;
}

common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx) {
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    if (llama_n_rs_seq(ctx) > 0) {
        COM_TRC("%s", "the context supports bounded partial sequence removal\n");
        return COMMON_CONTEXT_SEQ_RM_TYPE_RS;
    }

    common_context_seq_rm_type res = COMMON_CONTEXT_SEQ_RM_TYPE_PART;

    llama_memory_clear(mem, true);

    // FORK: a DFlash/DSpark draft cannot decode without target feature rows (llama-dflash.cpp "missing DFlash target
    // feature rows"), so the 2-token probe below always failed with -1, logged an ERROR at every startup and classed the
    // context NO. Upstream's draft accepts the probe and lands on PART. Skip only the decode; the removal test that
    // decides the answer still runs on this context's own memory (callers compare against FULL/RS only).
    if (llama_model_dflash_n_target_layers(llama_get_model(ctx)) > 0) {
        res = llama_memory_seq_rm(mem, 0, 1, -1) ? COMMON_CONTEXT_SEQ_RM_TYPE_PART : COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        COM_TRC("DFlash draft context: removal probe without a decode -> %s\n",
                res == COMMON_CONTEXT_SEQ_RM_TYPE_PART ? "PART" : "FULL");
        llama_memory_clear(mem, true);   // same cleanup as `done:` below (a goto here would skip initialisations)
        llama_synchronize(ctx);
        return res;
    }

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    // feed the probe in n_batch-sized pieces: a 2-token batch against -b 1 exceeds the context's
    // logical batch, which the legacy decoder refused (it aborted the whole server at startup, 2026-09-28).
    // Two cells end up in the cache either way, so the partial-removal test below is unchanged.
    // Each piece is decoded upstream's way: common_batch with a zero input state for input-state models.
    int ret = 0;
    {
        const size_t step = std::max<size_t>(1, std::min<size_t>(tmp.size(), llama_n_batch(ctx)));
        std::vector<float> zeros;
        for (size_t i = 0; i < tmp.size() && ret == 0; i += step) {
            common_batch batch = common_batch_get_one(ctx, tmp.data() + i, (int32_t) std::min(step, tmp.size() - i));
            common_batch_set_zero_state(batch, llama_get_model(ctx), zeros);
            ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        }
    }
    if (ret != 0) {
        COM_ERR("llama_process() failed: %d\n", ret);
        res = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        COM_TRC("%s", "the context does not support partial sequence removal\n");
        res = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx);

    return res;
}

static void common_context_seq_rm(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    if (!llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        GGML_ABORT("%s", string_format("failed to remove sequence %d with p0=%d, p1=%d\n", seq_id, p0, p1).c_str());
    }
}

static void common_context_seq_cp(llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
}

static void common_context_seq_add(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_add(mem, seq_id, p0, p1, delta);
}

void common_memory::init(llama_context * ctx_tgt, llama_context * ctx_dft) {
    this->ctx_tgt = ctx_tgt;
    this->ctx_dft = ctx_dft;
}

void common_memory::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    common_context_seq_rm(ctx_tgt, seq_id, p0, p1);
    if (ctx_dft) {
        common_context_seq_rm(ctx_dft, seq_id, p0, p1);
    }
}

void common_memory::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) const {
    common_context_seq_cp(ctx_tgt, seq_id_src, seq_id_dst, p0, p1);
    if (ctx_dft) {
        common_context_seq_cp(ctx_dft, seq_id_src, seq_id_dst, p0, p1);
    }
}

void common_memory::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) const {
    common_context_seq_add(ctx_tgt, seq_id, p0, p1, delta);
    if (ctx_dft) {
        common_context_seq_add(ctx_dft, seq_id, p0, p1, delta);
    }
}

void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora) {
    std::vector<llama_adapter_lora *> loras;
    std::vector<float> scales;

    for (auto & la: lora) {
        loras.push_back(la.ptr);
        scales.push_back(la.scale);
    }

    llama_set_adapters_lora(ctx, loras.data(), loras.size(), scales.data());
}

// --- FORK-ONLY, carried verbatim from ours in this merge: kv-cache/graph dtype and
//     batch-size helpers used by common_model_params_to_llama / common_context_params_to_llama
//     below (both hand-merged to keep the fork's extended field surface; see the progress
//     file for the full union-merge rationale). ---
// `what` names the FIELD. Without it this message is unactionable: fourteen call sites across
// seven distinct parameters feed this one function, and "Invalid cache type: " with an empty value
// reads as truncated output rather than as an empty input. Identifying the source of one such
// empty string required reading every caller and every writer -- that cost is the reason for this
// parameter. The value is quoted below for the same reason: '' is visibly empty, nothing is not.
static ggml_type kv_cache_type_from_str(const std::string & s, const char * what = "cache type") {
    if (s == "f32") {
        return GGML_TYPE_F32;
    }
    if (s == "f16") {
        return GGML_TYPE_F16;
    }
    if (s == "bf16") {
        return GGML_TYPE_BF16;
    }
    if (s == "f8_e4m3") { // mainline 202c1a229 spelling; must also be in common/arg.cpp kv_cache_types
        return GGML_TYPE_F8_E4M3;
    }
    if (s == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    if (s == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    if (s == "q4_1") {
        return GGML_TYPE_Q4_1;
    }
    if (s == "iq4_nl") {
        return GGML_TYPE_IQ4_NL;
    }
    if (s == "q5_0") {
        return GGML_TYPE_Q5_0;
    }
    if (s == "q5_1") {
        return GGML_TYPE_Q5_1;
    }
    if (s == "q6_0") {
        return GGML_TYPE_Q6_0;
    }
    if (s == "q8_KV") {
        return GGML_TYPE_Q8_KV;
    }
    // TurboQuant KV -- this is the SECOND parser; common/arg.cpp kv_cache_types is the first, and a name must be in
    // both (arg.cpp validates the flag, this one converts it at context creation).
    if (s == "turbo3") {
        return GGML_TYPE_TURBO3_0;
    }
    if (s == "turbo2") {
        return GGML_TYPE_TURBO2_0;
    }
    if (s == "turbo4") {
        return GGML_TYPE_TURBO4_0;
    }

    throw std::runtime_error(std::string("Invalid cache type for ") + what + ": '" + s + "'");
}

static std::pair<int, int> get_batch_ubatch(const common_params & params) {
    int n_batch = params.n_batch;
    int n_ubatch = params.n_ubatch;
    if (params.n_ctx > 0) {
        n_batch = std::min(n_batch, (int) params.n_ctx);
    }
    n_ubatch = std::min(n_batch, n_ubatch);
    return {n_batch, n_ubatch};
}

static ggml_type parse_ggml_type(const char * arg) {
    for (int j = 0; j < GGML_TYPE_COUNT; ++j) {
        auto type = ggml_type(j);
        const auto * name = ggml_type_name(type);
        if (name && strcmp(arg, name) == 0) {
            return type;
        }
    }
    return GGML_TYPE_COUNT;
}

static ggml_type ggml_type_from_str(const std::string & s) {
    if (s == "f32") {
        return GGML_TYPE_F32;
    }
    if (s == "f16") {
        return GGML_TYPE_F16;
    }
    if (s == "bf16") {
        return GGML_TYPE_BF16;
    }
    if (s == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    throw std::runtime_error("Invalid graph reduce type: " + s);
}

struct llama_model_params common_model_params_to_llama(common_params & params) {
    auto mparams = llama_model_default_params();

    // RETARGETED to device_spec 2026-09-13. This is the FOLLOW-UP OWED that include/llama.h:456
    // names by file and line: `mparams.devices` is upstream's NULL-terminated
    // ggml_backend_dev_t array, while common_params::devices is the fork's comma-separated
    // string (common.h:518, written by arg.cpp's -dev handler). The string form survives on
    // llama_model_params under the name `device_spec` (include/llama.h:459), and it is the one
    // src/llama.cpp:8407 actually parses -- it string_splits device_spec on ',' and matches the
    // pieces against buffer-type names, which is also how the fork's RPC "server|device"
    // addressing reaches the loader. mparams.devices stays at its default.
    if (!params.devices.empty()) {
        mparams.device_spec = params.devices.c_str();
    }

    if (params.n_gpu_layers != -1) {
        mparams.n_gpu_layers = params.n_gpu_layers;
    }
    mparams.mla             = params.mla_attn;
    mparams.dry_run         = params.dry_run;
    mparams.k_smooth_alpha  = params.k_smooth_alpha;
    mparams.nvfp4_precision = params.nvfp4_precision;
    mparams.rpc_servers     = params.rpc_servers.c_str();
    mparams.main_gpu        = params.main_gpu;
    mparams.max_gpu         = params.max_gpu;
    mparams.ncmoe           = params.ncmoe;
    mparams.fit             = params.fit;
    mparams.fit_margin      = params.fit_margin;
    mparams.worst_graph_tokens = params.worst_graph_tokens;
    mparams.type_k          = kv_cache_type_from_str(params.cache_type_k, "cache_type_k (--cache-type-k)");
    mparams.type_v          = kv_cache_type_from_str(params.cache_type_v, "cache_type_v (--cache-type-v)");
    mparams.idx_type_k      = kv_cache_type_from_str(params.indexer_cache_type_k, "indexer_cache_type_k (--indexer-cache-type-k)");
    mparams.type_k_first    = kv_cache_type_from_str(params.type_k_first, "type_k_first");
    mparams.type_k_last     = kv_cache_type_from_str(params.type_k_last, "type_k_last");
    mparams.type_v_first    = kv_cache_type_from_str(params.type_v_first, "type_v_first");
    mparams.type_v_last     = kv_cache_type_from_str(params.type_v_last, "type_v_last");
    if (!params.extra_output_type.empty()) {
        mparams.extra_output_type = parse_ggml_type(params.extra_output_type.c_str());
    }
    mparams.n_k_first       = params.n_k_first;
    mparams.n_k_last        = params.n_k_last;
    mparams.n_v_first       = params.n_v_first;
    mparams.n_v_last        = params.n_v_last;
    mparams.max_ctx_size    = params.n_ctx;
    // + n_cfg_guidance_slots (default 0, true no-op) + n_radix_cache_slots (only when
    // --radix-cache is set, default 0 contribution otherwise): see each field's comment in
    // common.h. Must match cparams.n_seq_max below.
    mparams.n_seq_max       = params.n_parallel + params.n_cfg_guidance_slots
                             + (params.radix_cache ? params.n_radix_cache_slots : 0)
                             + params.n_seq_decision;
    mparams.n_ubatch        = get_batch_ubatch(params).second;
    mparams.amb             = params.attn_max_batch;
    mparams.split_mode      = params.split_mode;
    mparams.lazy_mode       = params.lazy_mode;
    mparams.tensor_split    = params.tensor_split;
    // ADAPTED 2026-09-13. The failing side was the RIGHT-hand one: common_params dropped
    // use_mmap/use_mlock when it adopted `enum llama_load_mode load_mode` (common.h, upstream
    // e6dd0e29a "args: refactor mlock/mmap/directio into load-mode (#20834)"), which is why
    // `params.use_mmap` stopped resolving.
    //
    // CORRECTION to my first pass at this line, which assigned load_mode alone and said
    // llama_model_params "has no use_mmap/use_mlock members at all". It has all three
    // (include/llama.h:527-528 for the booleans, :502 for load_mode) -- and the booleans are
    // the ones that DO anything: src/llama.cpp:5103 passes params.use_mmap into
    // llama_model_loader and :5167 passes params.use_mlock on to the tensor loader, while
    // mparams.load_mode is READ NOWHERE in src/ (llama_load_mode_name/_from_str exist, but no
    // consumer). Handing over load_mode alone would have left mmap at its default and made
    // `--load-mode mlock` a silent no-op -- a build fix that quietly disables a flag.
    //
    // So decode the enum into the two booleans the loader reads, and pass the enum through as
    // well so nothing is dropped. Encoding per include/llama.h:334-339.
    mparams.load_mode       = params.load_mode;
    switch (params.load_mode) {
        case LLAMA_LOAD_MODE_NONE:       mparams.use_mmap = false; mparams.use_mlock = false; break;
        case LLAMA_LOAD_MODE_MMAP:       mparams.use_mmap = true;  mparams.use_mlock = false; break;
        case LLAMA_LOAD_MODE_MLOCK:      mparams.use_mmap = false; mparams.use_mlock = true;  break;
        case LLAMA_LOAD_MODE_MMAP_MLOCK: mparams.use_mmap = true;  mparams.use_mlock = true;  break;
        // The fork's loader has no DirectIO path, so DIO cannot be honoured; it is treated as
        // "no mmap", which is the half of it this loader can actually express. Not silent:
        // saying so here is the point, and the flag still parses rather than being rejected.
        case LLAMA_LOAD_MODE_DIRECT_IO:  mparams.use_mmap = false; mparams.use_mlock = false; break;
        // AUTO keeps whatever llama_model_default_params() chose (mmap on, mlock off), which is
        // exactly what "auto" meant before load_mode existed.
        case LLAMA_LOAD_MODE_AUTO:       break;
    }
    mparams.check_tensors   = params.check_tensors;
    mparams.repack_tensors  = params.repack_tensors;
    mparams.use_thp         = params.use_thp;
    mparams.validate_quants = params.validate_quants;
    mparams.merge_qkv       = params.merge_qkv;
    mparams.merge_up_gate_exps = params.merge_up_gate_exps;
    mparams.mtp             = params.has_mtp || params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);
    mparams.flash_attn      = params.flash_attn;
    mparams.defer_experts   = params.defer_experts;
    mparams.defer_ple       = params.defer_ple;
    mparams.defer_ple_auto  = params.defer_ple_auto;
    mparams.swa_compress    = params.swa_compress;

    // --- ported from upstream (theirs) in this merge ---
    mparams.use_extra_bufts = !params.no_extra_bufts;
    mparams.no_host         = params.no_host;
    mparams.progress_callback           = params.load_progress_callback;
    mparams.progress_callback_user_data = params.load_progress_callback_user_data;
    // upstream: mparams.load_mtp = contains(params.speculative.types, DRAFT_MTP). TRANSLATED:
    // header's common_params_speculative has no `.types` vector (only `.type`/`.stages`), so
    // this uses the fork's has_stage_type() over the resolved stage chain instead.
    // NOTE mparams.no_alloc (upstream) has no source field on common_params -- omitted, the
    // llama_model_default_params() default (false) applies; see progress-file escalation.

    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }
    if (params.tensor_buft_overrides.empty()) {
        mparams.tensor_buft_overrides = NULL;
    } else {
        GGML_ASSERT(params.tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern");
        mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    }
    if (!mparams.flash_attn && ggml_is_quantized(mparams.type_v)) {
        throw std::runtime_error("Quantized V cache cannot be used without flash attention");
    }
    if (!params.fit_margin_array.empty()) {
        GGML_ASSERT(params.fit_margin_array.size() % 2 == 0 && "Fit margin array does not have even number of elements");
        GGML_ASSERT(params.fit_margin_array[params.fit_margin_array.size()-2] == -1 && "Fit margin array is not correctly terminated");
        mparams.fit_margin_array = params.fit_margin_array.data();
    }
    mparams.load_mtp = params.load_mtp || params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);
    mparams.no_alloc = params.no_alloc;

    return mparams;
}

struct llama_context_params common_context_params_to_llama(const common_params & params) {
    auto cparams = llama_context_default_params();

    auto [n_batch, n_ubatch] = get_batch_ubatch(params);

    cparams.n_ctx             = params.n_ctx;
    // + n_cfg_guidance_slots (default 0, true no-op): see the field's comment in common.h. Must
    // match mparams.n_seq_max above -- both come from llama_model_default_params()/
    // llama_context_default_params() and llama-model.cpp's memory sizing reads cparams.n_seq_max,
    // so a mismatch here would under/over-size the KV/recurrent-state buffers built at model load.
    cparams.n_seq_max         = params.n_parallel + params.n_cfg_guidance_slots
                              + (params.radix_cache ? params.n_radix_cache_slots : 0)
                              + params.n_seq_decision;
    // ADOPTED 2026-09-13 alongside the two new common_params members; see the note there.
    // Without these two lines the members would be inert: llama_context_default_params sets
    // 0/1 (src/llama.cpp:8138-8139) and src/llama-context.cpp:299-301 resolves 0 to n_batch,
    // so speculative.cpp's dflash widening would never reach the context.
    cparams.n_outputs_max         = params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq;
    cparams.moe_cache_size        = params.moe_cache_size; // ik --moe-cache-mib (common/arg.cpp)
    cparams.moe_cache_cpu_share   = params.moe_cache_cpu_share;
    cparams.moe_cache_policy      = (enum llama_moe_cache_policy) params.moe_cache_policy;
    // --radix-cache: n_seq_max was just widened above by n_radix_cache_slots reserved ids. A
    // context-internal invariant (src/llama-context.cpp output_reserve(): local n_outputs_max =
    // max(n_outputs, n_seq_max()) must not exceed cparams.n_outputs_max, GGML_ASSERT at
    // src/llama-context.cpp:3022) assumes every live seq id might need an output slot.
    // MEASURED 2026-09-28: --radix-cache with its default 32 reserved ids plus a modest
    // --batch-size (params.n_outputs_max==0 resolves to n_batch downstream, src/llama-
    // context.cpp:475) tripped that assert on a real server start
    // (tools/server/tests/unit/test_radix_prefix_cache.py, --batch-size 32 --parallel 1 ->
    // n_seq_max 33 > n_outputs_max 32). Floor n_outputs_max at n_seq_max here, but ONLY when the
    // auto-derived value (n_batch, the same fallback the internal ctor would use) would
    // otherwise be smaller -- leaves every non-radix-cache server byte-identical to before.
    if (params.radix_cache || params.n_seq_decision > 0) {
        const uint32_t effective_n_outputs_max =
            params.n_outputs_max != 0 ? (uint32_t) params.n_outputs_max : (uint32_t) n_batch;
        if (cparams.n_seq_max > effective_n_outputs_max) {
            cparams.n_outputs_max = cparams.n_seq_max;
        }
    }
    cparams.n_batch           = n_batch;
    cparams.n_ubatch          = n_ubatch;
    // --- ported from upstream (theirs) in this merge ---
    // cpuparams is the CLI-wired thread source: arg.cpp populates params.cpuparams /
    // params.cpuparams_batch via postprocess_cpu_params(); ours' flat params.n_threads /
    // params.n_threads_batch are no longer set by anything after this merge, so those are
    // dropped in favour of theirs' cpuparams-based read.
    cparams.n_threads         = params.cpuparams.n_threads;
    cparams.n_threads_batch   = params.cpuparams_batch.n_threads == -1 ?
                                params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    // ik #2588-series CPU affinity: the backend side merged as-is. No CLI flag in this fork sets
    // params.cpu_affinity yet (ik's -cm/--cpu-mask name collides with this fork's upstream
    // -C/--cpu-mask in common/arg.cpp), so this stays nullptr until that port is decided.
    cparams.cpu_affinity      = params.cpu_affinity.empty() ? nullptr : params.cpu_affinity.data();
    cparams.n_cpu_affinity    = (int32_t) params.cpu_affinity.size();
    cparams.embeddings        = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.pooling_type      = params.pooling_type;
    cparams.attention_type    = params.attention_type;
    cparams.defrag_thold      = params.defrag_thold;
    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.flash_attn        = params.flash_attn;
    // upstream sets cparams.flash_attn_type = params.flash_attn_type directly; header's
    // common_params has no `.flash_attn_type` (bool `.flash_attn` stands, the header's own
    // union decision), so TRANSLATED from the bool to the tri-state enum:
    cparams.flash_attn_type   = params.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.no_perf           = params.no_perf;
    cparams.op_offload        = !params.no_op_offload;
    cparams.swa_full          = params.swa_full;
    cparams.kv_unified        = params.kv_unified;
    cparams.snapkv_cap        = params.snapkv_cap;        // S69 #22
    cparams.snapkv_window     = params.snapkv_window;
    cparams.h2o_heavy         = params.h2o_heavy;
    cparams.h2o_recent        = params.h2o_recent;
    // S69 #22: SnapKV and H2O score KV positions from the post-softmax attention tensor
    // ("kq_soft_max_ext-<il>", captured by snapkv_eval_callback). Flash attention fuses the softmax inside the
    // kernel and never materialises that tensor, so under the default -fa the eviction silently does nothing
    // (the capture is empty and the post-prefill block is a no-op). When either is requested, force the
    // non-flash path for this context and say so: the user opted into KV eviction over flash-attn speed.
    // DISABLE-REVERSIBLE: only when --snapkv-cap or --h2o-heavy is set (both default 0), flash attention is untouched otherwise.
    // PRIOR-ART-NONE: searched the tree for an in-kernel score capture under -fa (a flash kernel that returns the column sums); none exists here.
    // Ported from orphan 895ee0487 + cdbea913b, adapted to the tri-state flash_attn_type this tree carries.
    // fa_obs (row S69-OV-PICK-FA-OBSERVATION-WINDOW-ATTN-MASS-UNDER-FLASH-2026-10-06-01): flash attention is no longer forced off here.
    // llm_build_kqv adds a side branch on the FA path that materialises the same post-softmax tensor for the prune ubatch only;
    // the prune logs a warning when an arch's attention builder contributed no mass.
    cparams.kvflash_pool_tokens = params.kvflash_pool_tokens; // S69 #66
    cparams.offload_attn          = params.offload_attn;          // S69 #220
    cparams.offload_attn_page     = params.offload_attn_page;
    cparams.offload_attn_budget   = params.offload_attn_budget;
    cparams.offload_attn_selector = params.offload_attn_selector;
    cparams.offload_attn_sink     = params.offload_attn_sink;
    cparams.offload_attn_recent   = params.offload_attn_recent;
    cparams.offload_attn_dense_layers = params.offload_attn_dense_layers;
    std::snprintf(cparams.offload_attn_dense_spec, sizeof(cparams.offload_attn_dense_spec), "%s", params.offload_attn_dense_spec.c_str());
    cparams.offload_attn_head_reduce  = params.offload_attn_head_reduce;
    cparams.offload_attn_refresh  = params.offload_attn_refresh;
    cparams.offload_attn_scan     = params.offload_attn_scan;
    cparams.offload_attn_min_tokens = params.offload_attn_min_tokens;
    cparams.mla_attn          = params.mla_attn;
    cparams.attn_max_batch    = params.attn_max_batch;
    cparams.fused_moe_up_gate = params.fused_moe_up_gate;
    cparams.grouped_expert_routing = params.grouped_expert_routing;
    cparams.fused_up_gate     = params.fused_up_gate;
    cparams.fused_mmad        = params.fused_mmad;
    cparams.rope_cache        = params.rope_cache;
    cparams.graph_reuse       = params.graph_reuse;
    // S69-RPC-REUSE-ROOT-CAUSE-2026-09-28 round 2: cparams.graph_reuse gates the LEGACY-path graph
    // reuse (llama_context::can_reuse_graph(), src/llama.cpp:675) that "old-generation" models
    // (model.uses_legacy_graph_builder()==true, the DEFAULT per src/llama-model.h:1157 -- the
    // per-arch dispatcher that would flip it to false is unwired, see project memory
    // "models/ is WIRED but NOTHING calls the per-arch classes") actually run through --
    // COMPLETELY SEPARATE from process_ubatch's can_reuse()/graph_reuse_disable, which this fix's
    // earlier LLAMA_GRAPH_REUSE_DISABLE env var actually gates and which is DEAD CODE for such
    // models (confirmed this session: 0 process_ubatch calls observed via direct instrumentation on
    // stories260K, despite 16 RPC graph-compute calls happening). No CLI flag exists yet for
    // cparams.graph_reuse (params.graph_reuse, common/common.h:856, is a hardcoded-true struct
    // default with no arg.cpp wiring -- a still-open gap, separate from this bug). Added here,
    // diagnostic/testing only, reusing the same env var name for continuity: if unset, behavior is
    // UNCHANGED (params.graph_reuse's own default/CLI value, currently always true since nothing
    // sets it false).
    if (const char * v = getenv("LLAMA_GRAPH_REUSE_DISABLE")) {
        if (atoi(v) != 0) {
            cparams.graph_reuse = false;
        }
    }
    cparams.dsa               = params.dsa;
    cparams.fused_idx_topk    = params.fused_idx_topk;
    cparams.swa_compress      = params.swa_compress;
    cparams.dsv4_legacy_state = params.dsv4_legacy_state;
    cparams.dsa_top_k         = params.dsa_top_k;
    cparams.k_cache_hadamard  = params.k_cache_hadamard;
    cparams.v_cache_hadamard  = params.v_cache_hadamard;
    cparams.kv_fresh_chunk    = params.kv_fresh_chunk;
    cparams.split_mode_graph_scheduling = params.split_mode_graph_scheduling;
    cparams.scheduler_async   = params.scheduler_async;
    cparams.min_experts       = params.min_experts;
    cparams.thresh_experts    = params.thresh_experts;
    cparams.only_active_experts = params.only_active_exps;
    cparams.prefetch_experts  = params.prefetch_experts;
    cparams.prefetch_experts_threads = params.prefetch_experts_threads;
    cparams.max_extra_alloc   = params.max_extra_alloc_MiB;
    cparams.mtp               = params.has_mtp || params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);
    cparams.mtp_op_type       = MTP_OP_NONE;

    cparams.type_k = kv_cache_type_from_str(params.cache_type_k, "cache_type_k (--cache-type-k)");
    cparams.type_v = kv_cache_type_from_str(params.cache_type_v, "cache_type_v (--cache-type-v)");
    cparams.idx_type_k = kv_cache_type_from_str(params.indexer_cache_type_k, "indexer_cache_type_k (--indexer-cache-type-k)");
    cparams.type_reduce = ggml_type_from_str(params.reduce_type);
    cparams.type_graph_attn = ggml_type_from_str(params.graph_attn_precision);
    if (!cparams.flash_attn && ggml_is_quantized(cparams.type_v)) {
        throw std::runtime_error("Quantized V cache cannot be used without flash attention");
    }
    cparams.type_k_first    = kv_cache_type_from_str(params.type_k_first, "type_k_first");
    cparams.type_k_last     = kv_cache_type_from_str(params.type_k_last, "type_k_last");
    cparams.type_v_first    = kv_cache_type_from_str(params.type_v_first, "type_v_first");
    cparams.type_v_last     = kv_cache_type_from_str(params.type_v_last, "type_v_last");
    cparams.n_k_first       = params.n_k_first;
    cparams.n_k_last        = params.n_k_last;
    cparams.n_v_first       = params.n_v_first;
    cparams.n_v_last        = params.n_v_last;
    if (!cparams.flash_attn && ggml_is_quantized(cparams.type_v_first) && cparams.n_v_first > 0) {
        throw std::runtime_error("Quantized V cache cannot be used without flash attention");
    }
    if (!cparams.flash_attn && ggml_is_quantized(cparams.type_v_last) && cparams.n_v_last > 0) {
        throw std::runtime_error("Quantized V cache cannot be used without flash attention");
    }

    if (!params.offload_policy.empty()) cparams.offload_policy = (void *)&params.offload_policy;
    if (!params.cuda_params.empty()) cparams.cuda_params = (void *)params.cuda_params.data();

    // TriAttention KV scoring params (unified ik_llama build) -- RESTORED 2026-08-20.
    // NOTE the lifetime: tri_stats_path is a const char* into params.triattention_stats,
    // which must outlive the llama_context_params. It does here -- common_params owns it for
    // the life of the process -- and llama.cpp copies it into a std::string on the way in.
    cparams.tri_stats_path = params.triattention_stats.empty() ? nullptr
                                                              : params.triattention_stats.c_str();
    cparams.tri_budget     = params.tri_budget;
    cparams.tri_window     = params.tri_window;
    cparams.tri_interval   = params.tri_interval;
    cparams.tri_sink       = params.tri_sink;

    // CATS activation sparsity (S69-CATS-STAGE1-2026-09-28). Same c_str() lifetime note as
    // tri_stats_path just above: common_params owns the string for the life of the process.
    cparams.cats_sparsity        = params.cats_sparsity;
    cparams.cats_thresholds_file = params.cats_thresholds_file.empty() ? nullptr
                                                                       : params.cats_thresholds_file.c_str();
    cparams.cats_calibrate       = params.cats_calibrate;
    // MoE resident expert pool (S69-KT-EXPERT-POOL-2026-09-28). Same const-char* lifetime
    // note as tri_stats_path above: common_params outlives the llama_context_params.
    cparams.moe_pool_slots         = params.moe_pool_slots;
    cparams.moe_pool_inserts       = params.moe_pool_inserts;
    cparams.moe_pool_admit         = params.moe_pool_admit;
    cparams.moe_pool_cpu_only      = params.moe_pool_cpu_only;
    cparams.moe_stats_path         = params.moe_stats_path.empty() ? nullptr : params.moe_stats_path.c_str();
    cparams.moe_stats_mode         = params.moe_stats_mode.empty() ? nullptr : params.moe_stats_mode.c_str();
    cparams.moe_placement_strategy = params.moe_placement_strategy.empty() ? nullptr : params.moe_placement_strategy.c_str();
    cparams.moe_placement_stats    = params.moe_placement_stats.empty() ? nullptr : params.moe_placement_stats.c_str();
    cparams.moe_dynamic_update     = params.moe_dynamic_update;
    cparams.moe_dynamic_threshold  = params.moe_dynamic_threshold;
    cparams.ffd                    = params.ffd;
    cparams.ffd_block              = params.ffd_block;
    cparams.kivi_bits              = params.kivi_bits;
    cparams.kivi_group             = params.kivi_group;
    cparams.kivi_residual          = params.kivi_residual;
    cparams.ffd_delta              = params.ffd_delta;
    cparams.xattn_threshold        = params.xattn_threshold;
    cparams.xattn_stride           = params.xattn_stride;
    cparams.xattn_block            = params.xattn_block;
    cparams.kv_diversity_mass      = params.kv_diversity_mass;
    cparams.kv_diversity_block     = params.kv_diversity_block;
    cparams.kv_diversity_layers    = params.kv_diversity_layers;
    cparams.snapkv_rebase          = params.snapkv_rebase;

    // TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28). Same c_str() lifetime note.
    cparams.teal_sparsity        = params.teal_sparsity;
    cparams.teal_thresholds_file = params.teal_thresholds_file.empty() ? nullptr
                                                                        : params.teal_thresholds_file.c_str();
    cparams.teal_calibrate       = params.teal_calibrate;

    cparams.moe_cache_size = params.moe_cache_size;

    return cparams;
}

//
// Threadpool utils
//

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = params.priority;
    tpp.poll       = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

common_threadpools::~common_threadpools() {
    if (!free_fn) {
        return;
    }
    free_fn(threadpool);
    free_fn(threadpool_batch);
}

void common_threadpools::init(llama_context * ctx, const common_params & params) {
    GGML_ASSERT(!threadpool);
    GGML_ASSERT(!threadpool_batch);

    COM_INF("llama threadpool init, n_threads = %d\n", (int) params.cpuparams.n_threads);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        COM_WRN("%s", "no CPU backend found\n");
        return;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    // each pool needs to match the respective n_threads exactly
    // see: https://github.com/ggml-org/llama.cpp/pull/27138#issuecomment-5332307332
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            COM_WRN("batch threadpool create failed : n_threads %d\n", tpp_batch.n_threads);
            return;
        }

        // start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        COM_WRN("threadpool create failed : n_threads %d\n", tpp.n_threads);
        free_fn(threadpool_batch);
        threadpool_batch = nullptr;
        return;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);
}

//
// Vocab utils
//

std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

void common_vocab_token_table(const llama_vocab * vocab, std::vector<uint32_t> & lens, std::vector<uint8_t> & bytes) {
    const size_t vocab_size = llama_vocab_n_tokens(vocab);
    lens.assign(vocab_size, 0);
    bytes.clear();
    bytes.reserve(vocab_size * 8); // ~7 bytes per token is typical; only a reservation, the vector grows to what the vocab holds

    // One scratch buffer for a piece, grown to what llama_token_to_piece_vocab reports it needs: a too-small buffer returns
    // -(bytes required), so no piece length is assumed. `at` bytes at the front are left for the special-token marker. A negative return
    // that does not ask for more room than we offered is a real failure.
    std::vector<char> piece(256);
    auto render = [&](llama_token token, size_t at, bool special) -> int32_t {
        for (;;) {
            const int32_t n = llama_token_to_piece_vocab(vocab, token, piece.data() + at, (int32_t) (piece.size() - at), 0, special);
            if (n >= 0) {
                return n;
            }
            const size_t need = at + (size_t) (-(int64_t) n);
            if (need <= piece.size()) {
                GGML_ABORT("llama_token_to_piece failed\n");
            }
            piece.resize(need);
        }
    };

    for (size_t i = 0; i < vocab_size; i++) {
        const llama_token token = (llama_token) i;
        // The table must hold the bytes each token contributes MID-SEQUENCE. llama_detokenize of a single
        // token treats it as the FIRST token and, for a vocab with add_space_prefix (SentencePiece llama),
        // strips its leading space (llama-vocab.cpp detokenize: remove_space = add_space_prefix). Upstream
        // does the same; its test only uses a BPE vocab, where nothing is stripped. MEASURED 2026-09-28 on
        // stories260K (SPM): grammar "yes"|"no"|"maybe" produced " no" -- llguidance believed the SPM
        // space-marker token (U+2581 + "no") was "no". token_to_piece with lstrip = 0 keeps the leading space.
        int32_t size = render(token, 0, false);
        if (size == 0) {
            size = render(token, 1, true);
            if (size != 0) {
                piece[0] = '\xff';  // special token prefix marker
                size += 1;
            }
        }
        bytes.insert(bytes.end(), (const uint8_t *) piece.data(), (const uint8_t *) piece.data() + size);
        lens[i] = (uint32_t) size;
    }
}

std::string common_detokenize(const struct llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_detokenize(vocab, tokens, special);
}

std::string common_detokenize(const struct llama_vocab * vocab, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//
// Embedding utils
//

void common_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n){
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum  += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0) {
        if (sum1 == 0.0 && sum2 == 0.0) {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

//
// Control vector utils
//

static common_control_vector_data common_control_vector_load_one(const common_control_vector_load_info & load_info, int32_t n_layer) {
    common_control_vector_data result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(load_info.fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        COM_ERR("failed to load control vector file from %s\n", load_info.fname.c_str());
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        COM_WRN("no direction tensors found in %s\n", load_info.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            COM_ERR("invalid/unparsable direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            COM_ERR("invalid (zero) direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx > n_layer) {
            COM_ERR("direction tensor layer index %d exceeds the model layer count %d in %s\n", layer_idx, n_layer, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            COM_ERR("invalid (non-F32) direction tensor type in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            COM_ERR("invalid (non-1D) direction tensor shape in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            COM_ERR("direction tensor in %s does not match previous dimensions\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), (size_t) result.n_embd * layer_idx), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + (size_t) result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.n_embd == -1) {
        COM_WRN("skipping %s due to invalid direction tensors\n", load_info.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos, int32_t n_layer) {
    common_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = common_control_vector_load_one(info, n_layer);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            COM_ERR("control vectors in %s does not match previous dimensions\n", info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        COM_ERR("%s", "no valid control vector files passed\n");
        result.data.clear();
    }

    return result;
}

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    const int64_t ndata        = (tokens.size() - ne_datapoint - 1) / stride;
    ggml_opt_dataset_t result = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, ne_datapoint, ne_datapoint, ndata, /*ndata_shard =*/ 1);

    llama_token * data   = (llama_token *) ggml_opt_dataset_data(result)->data;
    llama_token * labels = (llama_token *) ggml_opt_dataset_labels(result)->data;

    for (int64_t idata = 0; idata < ndata; ++idata) {
        memcpy(data   + idata*ne_datapoint, tokens.data() + idata*stride + 0, ne_datapoint*sizeof(llama_token));
        memcpy(labels + idata*ne_datapoint, tokens.data() + idata*stride + 1, ne_datapoint*sizeof(llama_token));
    }

    return result;
}

ggml_opt_optimizer_params common_opt_lr_pars(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    const lr_opt &            d      = *(lr_opt *) userdata;
    result.adamw.alpha = result.sgd.alpha = d.get_lr(d.epoch);
    result.sgd.wd = result.adamw.wd = d.wd;
    return result;
}

// TODO make all command line args case-insensitive
static inline bool eq_case_insensitive(char const* a, char const* b) {
    return !
#if defined(_MSC_VER)
        _stricmp
#else
        strcasecmp
#endif // defined(_MSC_VER)
        (a, b);
}

enum ggml_opt_optimizer_type common_opt_get_optimizer(const char * n) {
    if (eq_case_insensitive("adamw", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    }
    if (eq_case_insensitive("sgd", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_SGD;
    }
    return GGML_OPT_OPTIMIZER_TYPE_COUNT;
}

// TODO simplify to use just log and exp
static float const k_log_2 = std::log(2.f);

void lr_opt::init() {
    if (lr_min > 0 && lr_min < lr0) {
        float nhalf = std::log(lr0 / lr_min) / k_log_2;
        float e     = epochs;
        if (decay_epochs > 0 && decay_epochs < e) {
            e = decay_epochs;
        } else {
            decay_epochs = e;
        }
        scale_epoch = nhalf / e;
    }
}

float lr_opt::get_lr(float epoch) const {
    float r = lr_min <= 0 ? lr0 :
        epoch >= decay_epochs ? lr_min :
        lr0 * std::pow(0.5f, epoch * scale_epoch);
    LOG_INF("epoch %.2g lr=%.2g\n", epoch, r);
    return r;
}

bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos) {
    common_batch batch(ctx);
    batch.add(last_token, pos, 0, true);

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("%s: failed to replay last token\n", __func__);
        return false;
    }
    return true;
}

common_batch::common_batch(llama_context * ctx) : batch(llama_batch_ext_init(ctx)) {
    const auto rope_type = llama_model_rope_type(llama_get_model(ctx));
    n_pos = rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? GGML_MROPE_SECTIONS : 1;
}

void common_batch::clear() {
    tokens.clear();
}

int32_t common_batch::add(llama_token id, llama_pos pos, llama_seq_id seq_id, bool output) {
    tokens.push_back({ id, { pos, 0, 0, 0 }, seq_id, output, { nullptr, 0, 0 }, { nullptr, 0, 0 }, {} });
    return size() - 1;
}

int32_t common_batch::add(llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool output) {
    GGML_ASSERT(!seq_ids.empty());

    const int32_t idx = add(id, pos, seq_ids[0], output);
    for (size_t s = 1; s < seq_ids.size(); ++s) {
        add_seq(idx, seq_ids[s]);
    }
    return idx;
}

bool common_batch::add_seq(int32_t idx, llama_seq_id seq_id) {
    if (idx < 0 || idx >= size()) {
        return false;
    }
    tokens[idx].seq_ids_extra.push_back(seq_id);
    return true;
}

bool common_batch::set_output(int32_t idx, bool value) {
    if (idx < 0 || idx >= size()) {
        return false;
    }
    tokens[idx].output = value;
    return true;
}

bool common_batch::set_embd(int32_t idx, llama_embd embd) {
    if (idx < 0 || idx >= size() || tokens[idx].embd.data != nullptr) {
        return false;
    }
    tokens[idx].embd = embd;
    return true;
}

bool common_batch::set_embd_state(int32_t idx, llama_embd state) {
    if (idx < 0 || idx >= size() || tokens[idx].state.data != nullptr) {
        return false;
    }
    tokens[idx].state = state;
    return true;
}

int32_t common_batch::add_embd(llama_embd embd, const llama_pos * pos, llama_seq_id seq_id, bool output) {
    token t = { LLAMA_TOKEN_NULL, { 0, 0, 0, 0 }, seq_id, output, embd, { nullptr, 0, 0 }, {} };
    for (int32_t j = 0; j < n_pos; ++j) {
        t.pos[j] = pos[j];
    }
    tokens.push_back(t);
    return size() - 1;
}

llama_batch_ext * common_batch::get_sub_batch(int32_t off, int32_t n) {
    GGML_ASSERT(batch && "common_batch was not initialized with a context");
    GGML_ASSERT(off >= 0 && n >= 0 && off + n <= size());

    llama_batch_ext * res = batch.get();
    llama_batch_ext_clear(res);

    for (int32_t i = off; i < off + n; ++i) {
        const token & t = tokens[i];

        int32_t idx;
        if (t.id != LLAMA_TOKEN_NULL) {
            idx = llama_batch_ext_add_token(res, t.seq_id, t.id);
            if (idx < 0) {
                GGML_ABORT("%s: failed to add token %d at index %d (error %d, n = %d)\n", __func__, t.id, i, idx, n);
            }
            llama_batch_ext_set_pos(res, idx, t.pos.data());
            if (t.embd.data && !llama_batch_ext_set_embd_token(res, idx, t.embd)) {
                GGML_ABORT("%s: failed to set the embedding of token %d at index %d\n", __func__, t.id, i);
            }
        } else {
            idx = llama_batch_ext_add_embd(res, t.seq_id, t.embd);
            if (idx < 0) {
                GGML_ABORT("%s: failed to add embedding at index %d (error %d, n = %d)\n", __func__, i, idx, n);
            }
            llama_batch_ext_set_pos(res, idx, t.pos.data());
        }
        GGML_ASSERT(idx == i - off);

        for (const llama_seq_id seq_id : t.seq_ids_extra) {
            if (!llama_batch_ext_add_seq(res, idx, seq_id)) {
                GGML_ABORT("%s: failed to add seq %d to the entry at index %d\n", __func__, seq_id, i);
            }
        }
        if (t.output) {
            llama_batch_ext_set_output_logits(res, idx, true);
        }
        if (t.state.data) {
            llama_batch_ext_set_embd_state(res, idx, t.state); // contexts without a state input ignore it
        }
        if (t.decision_order != 0) {
            llama_batch_ext_set_decision_order(res, idx, (llama_decision_order) t.decision_order);
        }
    }

    return res;
}

common_batch common_batch_get_one(llama_context * ctx, const llama_token * tokens, int32_t n_tokens) {
    common_batch batch(ctx);

    auto mem = llama_get_memory(ctx);
    llama_pos pos = llama_memory_seq_pos_max(mem, 0) + 1; // -1 + 1 == 0 when the memory is empty

    for (int32_t i = 0; i < n_tokens; ++i) {
        const bool output = i == n_tokens - 1;
        batch.add(tokens[i], pos, 0, output);
        pos++;
    }

    return batch;
}

common_batch common_batch_get_one(llama_context * ctx, const llama_tokens & tokens) {
    return common_batch_get_one(ctx, tokens.data(), (int32_t) tokens.size());
}

void common_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void common_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

bool common_prompt_batch_decode(
              struct llama_context * ctx,
                const llama_tokens & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state) {
    if (n_new == 0) {
        return true;
    }
    const int offset = all_tokens.size() - n_new;

    if (save_state && n_new > 1) {
        const int n_tokens_before_last = n_new - 1;

        GGML_ASSERT(n_new <= n_batch);

        // Decode all but the last token so we can save the memory state before decoding the last token.
        // This is done so we can restore the session state later and replay the last token.
        // Memory implementations in recurrent/hybrid models don't support removing tokens from their
        // memory, so we can't just remove the last token from the memory and replay the last token which
        // is the reason for this logic.
        llama_tokens prefix_tokens(all_tokens.begin() + offset, all_tokens.begin() + offset + n_tokens_before_last);
        common_batch batch_prefix = common_batch_get_one(ctx, prefix_tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch_prefix.get())) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_tokens_before_last;

        llama_state_save_file(ctx, state_path.data(), all_tokens.data(), all_tokens.size());
        COM_INF("saved session before last token to %s, n_new = %zu\n", state_path.data(), all_tokens.size());

        common_batch batch_last(ctx);
        batch_last.add(all_tokens.back(), n_past, 0, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch_last.get())) {
            COM_ERR("%s", "failed to eval last token\n");
            return false;
        }
        n_past++;
    } else {
        llama_tokens new_tokens(all_tokens.begin() + offset, all_tokens.begin() + offset + n_new);
        common_batch batch = common_batch_get_one(ctx, new_tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_new;
    }

    return true;
}

size_t common_prompt_checkpoint::size() const {
    return data_tgt.size() + data_dft.size() + data_spec.size();
}

bool common_prompt_checkpoint::empty() const {
    return data_tgt.empty();
}

void common_prompt_checkpoint::clear() {
    n_tokens = 0;
    cram.reset();
    cram_len[0] = cram_len[1] = cram_len[2] = 0;

    pos_min = 0;
    pos_max = 0;

    data_tgt.clear();
    data_dft.clear();
    data_spec.clear();
    disk.reset(); // the data is gone, so is any claim on a blob that held the old data
}

void common_prompt_checkpoint::update_pos(
        int64_t n_tokens,
        llama_pos pos_min,
        llama_pos pos_max) {
    this->n_tokens = n_tokens;
    this->pos_min  = pos_min;
    this->pos_max  = pos_max;
}

void common_prompt_checkpoint::update_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    disk.reset(); // new data: a tier blob written for the old data no longer describes it
    cram.reset(); // likewise a compressed store
    data_tgt.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_tgt.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

void common_prompt_checkpoint::update_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    disk.reset();
    cram.reset();
    data_dft.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_dft.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

bool common_prompt_checkpoint::load_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return true;
    }

    if (data_tgt.empty()) {
        return true;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_tgt.data(), data_tgt.size(), seq_id, flags);
    return n == data_tgt.size();
}

bool common_prompt_checkpoint::load_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return true;
    }

    if (data_dft.empty()) {
        return true;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_dft.data(), data_dft.size(), seq_id, flags);
    return n == data_dft.size();
}

// ---- context-checkpoint disk spill -------------------------------------------------------------
// File layout (little-endian host order, same machine only -- a spill file never outlives the server process):
//   u32 magic "CKS1", u32 version, i32 pos_min, i32 pos_max, i64 n_tokens,
//   then three records { u64 length, bytes } for data_tgt, data_dft, data_spec.
// The positions and n_tokens are written so reload() can refuse a file that does not belong to this entry.
static constexpr uint32_t COMMON_CKPT_SPILL_MAGIC   = 0x31534b43; // "CKS1"
static constexpr uint32_t COMMON_CKPT_SPILL_VERSION = 1;
static constexpr size_t   COMMON_CKPT_SPILL_HEADER  = 4 + 4 + sizeof(llama_pos) * 2 + sizeof(int64_t);

bool common_prompt_checkpoint::spill(const std::string & path) {
    if (data_tgt.empty() || path.empty()) {
        return false;
    }

    const uint32_t magic   = COMMON_CKPT_SPILL_MAGIC;
    const uint32_t version = COMMON_CKPT_SPILL_VERSION;
    const std::vector<uint8_t> * parts[3] = { &data_tgt, &data_dft, &data_spec };

    size_t expected = COMMON_CKPT_SPILL_HEADER;
    for (const auto * part : parts) {
        expected += sizeof(uint64_t) + part->size();
    }

    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) {
            return false;
        }
        f.write(reinterpret_cast<const char *>(&magic),   sizeof(magic));
        f.write(reinterpret_cast<const char *>(&version), sizeof(version));
        f.write(reinterpret_cast<const char *>(&pos_min), sizeof(pos_min));
        f.write(reinterpret_cast<const char *>(&pos_max), sizeof(pos_max));
        f.write(reinterpret_cast<const char *>(&n_tokens), sizeof(n_tokens));
        for (const auto * part : parts) {
            const uint64_t len = part->size();
            f.write(reinterpret_cast<const char *>(&len), sizeof(len));
            if (len > 0) {
                f.write(reinterpret_cast<const char *>(part->data()), (std::streamsize) len);
            }
        }
        f.flush();
        if (!f.good()) {
            f.close();
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return false;
        }
    }

    // verify what reached the disk before freeing the only in-memory copy
    std::error_code ec;
    const auto on_disk = std::filesystem::file_size(path, ec);
    if (ec || (size_t) on_disk != expected) {
        std::filesystem::remove(path, ec);
        return false;
    }

    spill_path = path;
    std::vector<uint8_t>().swap(data_tgt);
    std::vector<uint8_t>().swap(data_dft);
    std::vector<uint8_t>().swap(data_spec);
    return true;
}

// ---- context-checkpoint NVMe tier blobs (--cache-disk-dir) ------------------------------------------
// One tier blob = payload one: data_tgt; payload two: u64 len(data_dft), data_dft, u64 len(data_spec), data_spec.
// The positions and n_tokens stay in memory and go into the tag, so a blob is only handed back to the checkpoint it was
// written for. Checksumming, the size cap and the SSD write budget are the tier's (common/disk-tier.cpp).
int64_t common_prompt_checkpoint::tier_tag() const {
    const int64_t k[3] = { n_tokens, (int64_t) pos_min, (int64_t) pos_max };
    return (int64_t) common_disk_tier::checksum(k, sizeof(k), 0x63686b70ull /* "ckpt" */);
}

bool common_prompt_checkpoint::spill_tier(const std::shared_ptr<common_disk_tier> & tier) {
    if (!tier || data_tgt.empty()) {
        return false;
    }

    // already written (this checkpoint was reloaded from its blob): only free the memory, nothing is written again
    if (disk && disk->tier().get() == tier.get() && tier->has(disk->id())) {
        std::vector<uint8_t>().swap(data_tgt);
        std::vector<uint8_t>().swap(data_dft);
        std::vector<uint8_t>().swap(data_spec);
        return true;
    }
    disk.reset(); // a blob of another tier, or one the tier no longer has

    std::vector<uint8_t> packed;
    packed.reserve(tier_payload_b());
    const uint64_t len_dft = data_dft.size(), len_spec = data_spec.size();
    const auto * pl = reinterpret_cast<const uint8_t *>(&len_dft);
    packed.insert(packed.end(), pl, pl + sizeof(len_dft));
    packed.insert(packed.end(), data_dft.begin(), data_dft.end());
    pl = reinterpret_cast<const uint8_t *>(&len_spec);
    packed.insert(packed.end(), pl, pl + sizeof(len_spec));
    packed.insert(packed.end(), data_spec.begin(), data_spec.end());

    const uint64_t id = tier->put(data_tgt, packed, tier_tag(), /*is_ckpt=*/true);
    if (id == 0) {
        return false;
    }

    disk = std::make_shared<common_disk_blob>(tier, id);
    std::vector<uint8_t>().swap(data_tgt);
    std::vector<uint8_t>().swap(data_dft);
    std::vector<uint8_t>().swap(data_spec);
    return true;
}

// ---- CRAM software tier for checkpoints (--ctx-ckpt-cram-soft) ---------------------------------
static std::mutex g_ckpt_cram_mu;
static common_ckpt_cram_counters g_ckpt_cram;

common_ckpt_cram_counters common_ckpt_cram_get_counters() {
    std::lock_guard<std::mutex> lk(g_ckpt_cram_mu);
    return g_ckpt_cram;
}

void common_ckpt_cram_reset_counters() {
    std::lock_guard<std::mutex> lk(g_ckpt_cram_mu);
    g_ckpt_cram = {};
}

bool common_prompt_checkpoint::cram_compress() {
    if (data_tgt.empty()) {
        return false;
    }
    if (cram) { // reloaded earlier and not modified since (every mutator resets `cram`): the store still describes the data
        std::vector<uint8_t>().swap(data_tgt);
        std::vector<uint8_t>().swap(data_dft);
        std::vector<uint8_t>().swap(data_spec);
        return true;
    }
    std::vector<uint8_t> packed;
    packed.reserve(size());
    packed.insert(packed.end(), data_tgt.begin(),  data_tgt.end());
    packed.insert(packed.end(), data_dft.begin(),  data_dft.end());
    packed.insert(packed.end(), data_spec.begin(), data_spec.end());
    const uint64_t lens[3] = { data_tgt.size(), data_dft.size(), data_spec.size() };
    try {
        llama_cram_soft_params p;
        p.codec = LLAMA_CRAM_CODEC_PLANE_HUF;
        p.stride = 1;
        auto st = std::make_shared<llama_cram_soft_store>(packed.data(), packed.size(), p);
        const auto stt = st->stats();
        {
            std::lock_guard<std::mutex> lk(g_ckpt_cram_mu);
            g_ckpt_cram.stores++;
            g_ckpt_cram.raw_bytes    += stt.raw_bytes;
            g_ckpt_cram.stored_bytes += stt.compressed_bytes + stt.meta_bytes;
        }
        LOG_INF("%s: CRAM soft tier: checkpoint (pos_min = %d, pos_max = %d) %.3f MiB -> %.3f MiB (%llu/%llu blocks stored raw)\n", __func__,
                pos_min, pos_max, stt.raw_bytes / 1048576.0, (stt.compressed_bytes + stt.meta_bytes) / 1048576.0,
                (unsigned long long) stt.raw_blocks, (unsigned long long) stt.blocks);
        cram = std::move(st);
    } catch (const std::exception & e) {
        LOG_WRN("%s: CRAM soft tier failed (%s); the checkpoint stays in memory\n", __func__, e.what());
        return false;
    }
    for (int i = 0; i < 3; i++) {
        cram_len[i] = lens[i];
    }
    std::vector<uint8_t>().swap(data_tgt);
    std::vector<uint8_t>().swap(data_dft);
    std::vector<uint8_t>().swap(data_spec);
    return true;
}

bool common_prompt_checkpoint::reload() {
    if (!spilled()) {
        return false;
    }

    if (cram) {
        const uint64_t total = cram_len[0] + cram_len[1] + cram_len[2];
        std::vector<uint8_t> buf(total);
        try {
            if (cram->read(0, total, buf.data()) != total) {
                throw std::runtime_error("short read");
            }
        } catch (const std::exception & e) {
            LOG_WRN("%s: CRAM soft tier read failed (%s); dropping the checkpoint data\n", __func__, e.what());
            cram.reset();
            return false;
        }
        data_tgt.assign(buf.begin(), buf.begin() + cram_len[0]);
        data_dft.assign(buf.begin() + cram_len[0], buf.begin() + cram_len[0] + cram_len[1]);
        data_spec.assign(buf.begin() + cram_len[0] + cram_len[1], buf.end());
        {
            std::lock_guard<std::mutex> lk(g_ckpt_cram_mu);
            g_ckpt_cram.reloads++;
            g_ckpt_cram.read_bytes += total;
        }
        LOG_INF("%s: CRAM soft tier: checkpoint (pos_min = %d, pos_max = %d) read back, %llu bytes\n", __func__, pos_min, pos_max, (unsigned long long) total);
        return true; // the store stays, so compressing this checkpoint again is a no-op for the caller to repeat
    }

    if (disk) {
        // PROGRESS-OK: single bounded read of one blob, no pipeline phases
        std::vector<uint8_t> a, packed;
        bool ok = disk->tier() && disk->tier()->get(disk->id(), tier_tag(), a, packed);
        std::vector<uint8_t> dft, spec;
        if (ok) {
            // the tier already verified checksum, lengths and tag; this checks that the packed vector is what spill_tier wrote
            uint64_t len_dft = 0, len_spec = 0;
            ok = packed.size() >= 2 * sizeof(uint64_t);
            if (ok) {
                std::memcpy(&len_dft, packed.data(), sizeof(len_dft));
                ok = len_dft <= packed.size() - 2 * sizeof(uint64_t);
            }
            if (ok) {
                std::memcpy(&len_spec, packed.data() + sizeof(len_dft) + len_dft, sizeof(len_spec));
                ok = packed.size() == 2 * sizeof(uint64_t) + len_dft + len_spec;
            }
            if (ok) {
                dft.assign(packed.begin() + sizeof(len_dft), packed.begin() + sizeof(len_dft) + len_dft);
                spec.assign(packed.end() - len_spec, packed.end());
            }
        }
        if (!ok) {
            disk.reset(); // the tier already dropped a blob that failed its checks
            data_tgt.clear();
            data_dft.clear();
            data_spec.clear();
            return false;
        }
        data_tgt  = std::move(a);
        data_dft  = std::move(dft);
        data_spec = std::move(spec);
        return true; // the blob stays in the tier while any copy of this checkpoint references it
    }

    return reload_file(/*remove_on_fail=*/true);
}

bool common_prompt_checkpoint::adopt_spill_file() {
    if (!data_tgt.empty() || spill_path.empty()) {
        return false;
    }
    // the file is somebody else's (the slot's own checkpoint wrote it): read it, never delete it, and let go of the path either way
    const bool ok = reload_file(/*remove_on_fail=*/false);
    spill_path.clear();
    return ok;
}

bool common_prompt_checkpoint::reload_file(bool remove_on_fail) {
    std::ifstream f;
    auto fail = [this, &f, remove_on_fail]() {
        f.close(); // Windows cannot delete a file that is still open
        if (remove_on_fail) {
            remove_spill_file();
        }
        spill_path.clear();
        data_tgt.clear();
        data_dft.clear();
        data_spec.clear();
        return false;
    };

    std::error_code ec;
    const auto on_disk = std::filesystem::file_size(spill_path, ec);
    if (ec || (size_t) on_disk < COMMON_CKPT_SPILL_HEADER + 3 * sizeof(uint64_t)) {
        return fail();
    }

    f.open(spill_path, std::ios::binary);
    if (!f) {
        return fail();
    }

    uint32_t  magic = 0, version = 0;
    llama_pos pmin = 0, pmax = 0;
    int64_t   ntok = 0;
    f.read(reinterpret_cast<char *>(&magic),   sizeof(magic));
    f.read(reinterpret_cast<char *>(&version), sizeof(version));
    f.read(reinterpret_cast<char *>(&pmin),    sizeof(pmin));
    f.read(reinterpret_cast<char *>(&pmax),    sizeof(pmax));
    f.read(reinterpret_cast<char *>(&ntok),    sizeof(ntok));
    if (!f || magic != COMMON_CKPT_SPILL_MAGIC || version != COMMON_CKPT_SPILL_VERSION ||
            pmin != pos_min || pmax != pos_max || ntok != n_tokens) {
        return fail();
    }

    std::vector<uint8_t> loaded[3];
    size_t remaining = (size_t) on_disk - COMMON_CKPT_SPILL_HEADER;
    for (auto & part : loaded) {
        uint64_t len = 0;
        if (remaining < sizeof(len)) {
            return fail();
        }
        f.read(reinterpret_cast<char *>(&len), sizeof(len));
        remaining -= sizeof(len);
        // a length larger than what is left in the file is corruption, not an allocation request
        if (!f || len > remaining) {
            return fail();
        }
        part.resize((size_t) len);
        if (len > 0) {
            f.read(reinterpret_cast<char *>(part.data()), (std::streamsize) len);
            if (!f) {
                return fail();
            }
        }
        remaining -= (size_t) len;
    }
    // the target data is the "resident" marker; a checkpoint with none is not usable
    if (remaining != 0 || loaded[0].empty()) {
        return fail();
    }

    data_tgt  = std::move(loaded[0]);
    data_dft  = std::move(loaded[1]);
    data_spec = std::move(loaded[2]);
    return true; // the file stays until the entry is dropped; a later spill() rewrites it
}

void common_prompt_checkpoint::remove_spill_file() {
    if (!spill_path.empty()) {
        std::error_code ec;
        std::filesystem::remove(spill_path, ec);
    }
}

void common_prompt_checkpoint::clear_tgt() {
    cram.reset();
    data_tgt.clear();
}

void common_prompt_checkpoint::clear_dft() {
    cram.reset();
    data_dft.clear();
    data_spec.clear();
}


// ============================================================================================
// RESTORED 2026-09-13 from 279b4bfaf^1 (the merge's first parent, our clean pre-merge side).
//
// These bodies were dropped when the merge replaced this file. The loss produced no
// compile error anywhere: a missing definition is a LINK error, and nothing linked
// common/ because common/ itself did not compile. string_is_found surfaced only once
// src/llama-context.cpp was wired and cleared enough of the link surface for it to
// reach the top of the undefined list.
//
// Taken VERBATIM from 279b4bfaf^1 -- NOT from common.cpp.pre-fmt-all, which is a
// mid-merge file with 47 conflict regions whose '=======' line cuts string_is_found's
// body before its closing brace.
// ============================================================================================

// string_is_found -- linker reports it undefined at 46 references; called common/sampling.cpp:979, :983
bool string_is_found(const std::string& window, const std::string& str, size_t& pos) {
    if (str.empty()) {
        return false;
    }
    pos = window.find(str);
    return pos != std::string::npos;
}

// string_unescape -- called common/sampling.cpp:980 (x2), :984 (x2)
std::string string_unescape(const std::string& str) {
    std::string result;
    result.reserve(2 * str.length());
    for (const auto c: str) {
        switch (c) {
        case '\n':
            result.append("\\n");
            break;
        case '\t':
            result.append("\\t");
            break;
        case '\r':
            result.append("\\r");
            break;
        default:
            result.append(1, c);
            break;
        }
    }
    return result;
}

// string_lower -- called examples/spec-bench/spec-bench.cpp:403
std::string string_lower(const std::string& str) {
    std::string result = str;
    for (char& c : result) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c + ('a' - 'A'));
        }
    }
    return result;
}

// string_extract -- declared common/common.h:1298, defined nowhere
std::vector<std::string> string_extract(const std::string& str, const char c, std::vector<size_t>& posi) {
    std::vector<std::string> extracts;
    auto pos = str.find(c);
    size_t count = 0;
    while (pos != std::string::npos) {
        if (count % 2 == 0) {
            // opening c
            posi.push_back(pos);
            ++count;
        } else {
            // closing c must be unescaped
            auto esc_pos = pos;
            size_t n_esc = 0;
            while ((esc_pos > 0) && (str[--esc_pos] == '\\')) {
                ++n_esc;
            }
            if (n_esc % 2 == 0) {
                extracts.push_back(str.substr(posi.back() + 1, pos - posi.back() - 1));
                string_process_escapes(extracts.back());
                posi.push_back(pos);
                ++count;
            }
        }
        pos = str.find(c, pos + 1);
    }
    return extracts;
}

// parse_buft_overrides -- called examples/llama-bench/llama-bench.cpp:426, :887
// parse_buft_overrides is NOT restored here. examples/llama-bench/llama-bench.cpp:425 defines its own
// copy inside an anonymous namespace, so this file never needed one -- and two bodies
// of one function is the defect just cleared out of src/llama.cpp. My call-site
// scanner read that definition as a CALL (it tested only for a ';' after the closing
// paren, so a definition's '{' fell through to "call"), which is what put it on the
// restore list. llama-bench's copy uses the removed ggml_backend_reg_* API and is
// fixed IN PLACE instead.


// Expiring logit bias config parser (--expiring-logit-bias-file). The sampler that consumes elb_params is live
// (sampling.cpp common_expiring_logit_bias_apply/_accept), but this parser AND its flag were removed by the upstream
// merge 279b4bfaf, leaving no way to set elb_params. Restored 2026-09-27 verbatim from common.cpp @ 35f0cac75
// (only LLAMA_LOG_DEBUG -> LOG_DBG). Row S69-UPSTREAM-MERGE-DROPPED-166-FORK-SERVER-CLI-FLAGS-2026-09-27-01.
void argparse_expiring_logit_bias(const std::string& content, common_params_sampling& sparams) {
    auto elb_params = sparams.elb_params;
    elb_params.push_back({ { }, "", "" });
    auto entries = elb_params[0].entries;

    const auto lines = string_split(content, "\n");
    for (size_t i = 0; i < lines.size(); ++i) {
        auto line = string_strip(lines[i]);
        const char c0 = line.empty() ? '#' : line[0];
        if (c0 == '#') {
            LOG_DBG("%s: line %zu: comment or empty\n", __func__, i);
            continue;   // next line
        }

        // (... "EXTRACT" ... "EXTRACT" ...)
        std::vector<size_t> qq_posi = { 0 };
        auto extracts = string_extract(line, '"', qq_posi);
        qq_posi.push_back(std::string::npos);
        for (int32_t j = 0; j < int32_t(qq_posi.size()) - 1; j += 2) {
            const auto pnd_pos = line.find('#', qq_posi[j]);
            if (pnd_pos < qq_posi[j + 1]) {
                LOG_DBG("%s: line %zu: inline comment @ %zu\n", __func__, i, pnd_pos);
                line = string_strip(line.substr(0, pnd_pos));
                qq_posi.resize(j + 2);
                qq_posi.back() = std::string::npos;
                extracts.resize(j / 2);
                break;
            }
        }
        const auto last_qq_pos = qq_posi[qq_posi.size() - 2];

        auto n_char = line.length();
        const char cE = line[n_char - 1];

        LOG_DBG("%s: line %zu: %s\n", __func__, i, line.c_str());
        if ('(' == c0 && cE == ')') {
            const bool is_nested = '(' == line[1] && line[n_char - 2] == ')';
            if (is_nested) {
                if (n_char == 4) {
                    // (())
                    entries.clear();
                    LOG_DBG("%s: line %zu: persistent entry clear\n", __func__, i);
                    continue;   // next line
                }
                n_char -= 2;
                line = line.substr(1, n_char);
                LOG_DBG("%s: line %zu: persistent entry\n", __func__, i);
            }

            // (DURATION : ...)
            int32_t duration = is_nested ? -1 : 1;
            const auto cln_pos = line.find(':');
            if ((cln_pos != std::string::npos) && (1 < cln_pos) && (cln_pos < qq_posi[1])) {
                duration = std::stoi(line.substr(1, cln_pos - 1));
            }
            if (duration == 0) {
                LOG_DBG("%s: line %zu: invalid duration\n", __func__, i);
                continue;   // next line
            }

            #undef X
            #define X(T, MEMBER, DV, PRECAST) #MEMBER,
            static const std::vector<std::string> names = { X_COMMON_PARAMS_SAMPLING };

            std::vector<float> addsubs(names.size(), 0.0f);
            bool is_epb = false;

            // (... : SPARAM ...)
            const auto window = line.substr(last_qq_pos + 1);
            for (int j = 0; j < names.size(); ++j) {
                const auto& name = names[j];
                auto pos = window.find(name);
                if (pos != std::string::npos) {
                    pos += name.length();
                    auto next_pos = window.find(",", pos + 1);
                    if (next_pos == std::string::npos) {
                        next_pos = n_char - 1;
                    }
                    auto sub = string_strip(window.substr(pos, next_pos - pos));
                    if (sub[0] == '~') {
                        addsubs[j] += std::stof(sub.substr(1));
                        is_epb = true;
                        LOG_DBG("%s: line %zu: bias = %f\n", __func__, i, addsubs[j]);
                    }
                }
            }

            auto& phrases = extracts;
            if (phrases.empty()) {
                if (is_epb) {
                    phrases.push_back("");
                } else {
                    continue;   // next line
                }
            }

            int32_t max_keyword_len = 0;
            const auto n_phrase = phrases.size();
            std::vector<float> biases;
            bool is_range = false;

            if (!is_epb) {
                // (... : BIAS ...)
                const auto cln_rpos = line.rfind(':');
                auto sub = line.substr(cln_rpos + 1, n_char - cln_rpos - 2);
                if (sub.find("~") != std::string::npos) {
                    // (... : BIAS ~ BIAS)
                    const auto splits = string_split<std::string>(sub, '~');
                    biases.push_back(std::stof(splits.front()));
                    LOG_DBG("%s: line %zu: logit bias = %f\n", __func__, i, biases.back());
                    biases.push_back(std::stof(splits.back()));
                    LOG_DBG("%s: line %zu: logit bias = %f\n", __func__, i, biases.back());
                    is_range = true;
                } else {
                    // (... : BIAS, BIAS, ..., BIAS)
                    for (const auto& split: string_split<std::string>(sub, ',')) {
                        if (!split.empty()) {
                            biases.push_back(std::stof(split));
                            LOG_DBG("%s: line %zu: logit bias = %f\n", __func__, i, biases.back());
                        }
                    }
                }
                if (biases.empty()) {
                    continue;   // next line
                }
            } else {
                for (const auto& keyword: phrases) {
                    LOG_DBG("%s: line %zu: keyword = \"%s\"\n", __func__, i, keyword.c_str());
                    max_keyword_len = std::max(SSIZE(keyword), max_keyword_len);
                }
                LOG_DBG("%s: line %zu: max_keyword_len = %d\n", __func__, i, max_keyword_len);
            }

            common_params_sampling::elb_param::elb_entry entry = {
                max_keyword_len,
                std::vector<int32_t>(n_phrase, 0),
                std::move(addsubs),
                std::vector<bool>(n_phrase, false),
                std::move(phrases),
                std::move(biases),
                duration,
                is_range
            };
            if (is_nested) {
                entries.push_back(entry);
            }
            elb_params.back().entries.push_back(std::move(entry));
            continue;   // next line
        }

        if (last_qq_pos > 0) {
            elb_params.back().op = string_strip(line.substr(last_qq_pos + 1));
        }

        auto& exitwords = extracts;
        if (exitwords.empty()) {
            string_process_escapes(line);
            exitwords.push_back(std::move(line));
        }

        // maybe support multiple exitwords in future
        elb_params.back().exitword = std::move(exitwords[0]);

        elb_params.push_back({ entries, "", "" });
    }

    sparams.elb_params = std::move(elb_params);
}
