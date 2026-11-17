#pragma once

#include "llama.h"
#include "llama-impl.h"
#include "llama-expert-io.h"
#include "llama-mmap.h"
#include "llama-arch.h"
#include "llama-hparams.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <map>
#include <array>

// "glm5-next" names two enums (see get_arch()): fork GLM5NEXT_DASHED and upstream GLM5_NEXT. LLAMA_GLM5NEXT_IMPL=upstream picks the upstream
// builder for the same file. The choice must be made in BOTH places that read the architecture -- llama_peek_architecture (llama.cpp, picks the
// model OBJECT's type: GLM5_NEXT is a src/models class) and the loader's get_arch() -- or the legacy loader meets GLM5_NEXT and throws
// "unknown architecture" (measured 2026-11-09). Default (unset / other value): the fork path.
static inline llm_arch llm_glm5next_impl_resolve(llm_arch a) {
    if (a == LLM_ARCH_GLM5NEXT_DASHED) {
        const char * impl = std::getenv("LLAMA_GLM5NEXT_IMPL");
        return (impl != nullptr && std::string(impl) == "upstream") ? LLM_ARCH_GLM5_NEXT : LLM_ARCH_GLM5NEXT;
    }
    return a;
}

// llama_buf_map: file-index -> the backend buffers covering that file's mmap region.
// ADOPTED from unslothai/llama.cpp PR #152 ("map each contiguous run of a context's tensors,
// not one span over all of them"): a foreign tensor sitting between this context's tensors
// splits the mapping, so one file index can own SEVERAL buffers. This is the type CONFIRMED
// live: src/llama.cpp's llama_model_load() (the actual body behind the public
// llama_model_load_from_file, verified by reading its call chain) declares `llama_buf_map bufs`
// and does `bufs[idx].push_back(buf)`, which only compiles against a vector-valued map.
// Upstream's own single-buffer-per-index alias is NOT carried as a competing type under this
// same name -- see the loader.cpp exclusion note for why that side is not wired to anything.
using llama_buf_map = std::unordered_map<uint32_t, std::vector<ggml_backend_buffer_t>>;
// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

// COMPATIBILITY-INVESTIGATED: read gguf.h (int64_t key_id), llama.h (llama_ftype_name/llama_load_mode
// already declared), src/llama.cpp call chain for llama_model_load_from_file, src/llama-quant.cpp,
// src/llama-quantize.cpp, src/llama-hparams.cpp, src/llama-load-tensors.cpp (6367 lines) to
// determine which loader surface each already-resolved consumer actually needs, before choosing
// what to carry here. See the per-field notes below for the specific evidence each decision rests on.
// Declaration only -- defined once in llama-model-loader.cpp (see the comment there). The header
// used to carry a `static` inline body too; that gave every including TU its own copy AND now
// collides with the .cpp's non-static definition (redefinition). Finishing the half-applied move.
const char * llama_file_version_name(llama_fver version);

struct llama_layer;

struct llama_model_loader {
    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;
    // names counted in n_created, and the ones counted more than once: a "too many tensors created" error then NAMES the
    // tensors created twice instead of only reporting the two counts (added 2026-11-05 for the LiLiCorr class load, 103 vs 108).
    std::set<std::string> created_names;
    std::vector<std::string> created_twice;
    int n_skipped = 0;
    size_t size_skipped = 0;

    int64_t n_elements = 0;
    size_t  n_bytes    = 0;

    int     ncmoe      = 0;

    bool use_mmap = false;
    bool check_tensors;
    bool repack_tensors = false;
    bool use_thp = false;
    bool merge_qkv = false;
    bool merge_up_gate_exps = false;
    bool defer_experts = false;
    // adopted from upstream in this merge. Not reachable from a live constructor path today (see
    // loader.cpp's exclusion note); kept so the field upstream's declared-but-undefined
    // llama_model_load_from_splits/llama_model_init_from_user (include/llama.h) can eventually
    // read exists, without pretending this merge implemented that path. Always false today.
    bool no_alloc = false;
    bool defer_ple = false;
    // --lazy-mode auto (llama_model_params::defer_ple_auto): a tensor flagged TENSOR_READ_LAZY joins the deferred index
    // only above 4 GiB. Set by llama_model_load next to defer_ple.
    bool defer_ple_auto = false;
    // names of tensors the arch created with TENSOR_READ_LAZY (upstream flag; src/llama-model-loader.cpp:1352 @ unsloth master)
    std::vector<std::string> lazy_flagged;
    // Explicit --no-mmap with deferred tables: copy dense weights out, alias deferred only.
    bool defer_copy_dense = false;
    std::vector<ggml_backend_buffer_type_t> integrated_bufts;

    llama_files files;
    // Split number (0-based, as in LLM_KV_SPLIT_NO) of files[i]. llama_tensor_weight::idx stays the
    // index into files/mappings; this maps it back to the split it came from. Identity unless the
    // loader was given a partial split set (only_splits, see the constructor), where skipped splits
    // leave no entry in files. split_no() falls back to identity for loaders that never fill it.
    std::vector<uint16_t> file_split_no;
    uint16_t split_no(uint16_t file_idx) const {
        return file_idx < file_split_no.size() ? file_split_no[file_idx] : file_idx;
    }
    // true when only_splits was given and at least one split of the model was not opened
    bool partial_splits = false;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;
        std::vector<float> nvfp4_scale;
        bool mxfp4_repack = false; // MXFP4 on disk, NVFP4_KS in memory (native W4A4); see import_nvfp4()

        // MEASURED (2026-09-07, this merge): upstream's side of this same constructor adds a
        // `tensor_idx < 0` check before computing offs -- a real robustness improvement (a
        // negative index would otherwise flow into gguf_get_tensor_offset and corrupt offs
        // rather than fail loudly). Carried below. Fork's explicit `name` parameter (as opposed
        // to reading it back off `tensor`) and its richer stderr diagnostic on an out-of-bounds
        // tensor are kept too -- both sides' real value, no capability dropped.
        llama_tensor_weight(const llama_file * file, uint16_t idx, const char * name, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx, name);
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", name));
            }
            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);

            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                auto data_offset = gguf_get_data_offset(gguf_ctx);
                auto tensor_offset = gguf_get_tensor_offset(gguf_ctx, tensor_idx);
                fprintf(stderr, "Error while loading tensor %s: offs = %zu (%zu, %zu), size: %zu, file size: %zu\n", name,
                        offs, data_offset, tensor_offset, ggml_nbytes(tensor), file->size());
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", name));
            }
        }
    };
    // `weights`: fork's own ordering, indexed access (get_tensor_name(i)/get_weight(i)) -- confirmed
    // live: src/llama.cpp's llama_model_load() (the real body behind llama_model_load_from_file)
    // iterates `for (int i = 0; i < ml.n_tensors; ++i) ml.get_weight(i)`.
    std::vector<llama_tensor_weight> weights;
    // `weights_map`: upstream's by-name index -- confirmed live by a DIFFERENT consumer,
    // src/llama-quant.cpp (already resolved, not part of this conflict cluster), which does
    // `for (const auto & it : ml.weights_map)` and `gguf_set_kv(ctx_out.get(), ml.metadata)`.
    // A plain std::map (not upstream's weight_name_comparer-sorted one) suffices: llama-quant.cpp
    // re-sorts its own working copy by idx/offs for keep_split and does not depend on this map's
    // iteration order. Populated in parallel with `weights` in the constructor, from the SAME
    // llama_tensor_weight value, so the two containers can never disagree.
    std::map<std::string, llama_tensor_weight> weights_map;

    std::unordered_map<std::string, struct llama_model_kv_override> kv_overrides;
    const llama_model_tensor_buft_override * tensor_buft_overrides;

    // `meta` (fork's name) and `metadata` (upstream's name) are the SAME gguf_context*, set
    // together in the constructor. Both names have live callers: `meta` throughout this file and
    // in src/llama-hparams.cpp/src/llama-quantize.cpp; `metadata` in src/llama-quant.cpp. Same
    // "one struct, both spellings" pattern include/llama.h already applies to kv_overrides/imatrix.
    gguf_context * meta = NULL;
    gguf_context * metadata = NULL;
    std::vector<ggml_context *> contexts;

    // UNIT 1 (S69-PORT-MODEL-INIT-FROM-USER-AND-FILE-PTR-LOADER-2026-09-28-01). `owns_meta`: false
    // when `meta` was supplied by the CALLER (the metadata-only ctor below) rather than opened by
    // this loader from a path/FILE* -- the destructor must not gguf_free() memory it does not own.
    // SOURCE-READ: upstream llama-model-loader.h:130-131 ("gguf_context_ptr metadata_ptr;" /
    // "gguf_context * metadata; // either metadata_ptr.get() or externally set") -- upstream's
    // metadata_ptr (an OWNING unique_ptr) is left empty on the externally-supplied path so its own
    // destructor frees nothing; this fork's `meta` is a raw pointer freed unconditionally in
    // ~llama_model_loader(), so `owns_meta` reproduces the same non-ownership by hand. Defaults to
    // true, so the pre-existing fname-based ctor (below) is UNCHANGED: it always owns what it opens.
    bool owns_meta = true;

    // Delivers tensor BYTES at load time instead of reading them from `files` -- set only by the
    // metadata-only ctor below (never by the pre-existing fname ctor, so every existing caller's
    // load_data_for()/load_all_data() path is unaffected: both are gated on this being non-null).
    llama_model_set_tensor_data_t set_tensor_data    = nullptr;
    void *                        set_tensor_data_ud = nullptr;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);
    mutable bool arch_resolved = false;
    mutable llm_arch resolved_arch = LLM_ARCH_UNKNOWN;
    llama_expert_tensor_index expert_tensor_index;
    llama_expert_tensor_index ple_tensor_index;

    // Ported from upstream 2026-09-08: load_data_for()/load_all_data() track cumulative progress
    // and per-mapping (offset, length) ranges already faulted in across these three fields. The
    // .cpp side already used them (24 sites); only the member declarations were missing from this
    // header, so every use was "undeclared identifier".
    size_t size_done = 0;
    size_t size_data = 0;
    std::vector<std::pair<size_t, size_t>> mmaps_used;

    llama_model_loader(const std::string & fname, int ncmoe, bool use_mmap, bool check_tensors, bool repack_tensors, bool use_thp,
            bool merge_qkv, bool merge_up_gate_exps, bool defer_experts,
            const llama_model_kv_override * param_overrides_p,
            const llama_model_tensor_buft_override * param_tensor_buft_overrides_p,
            // Optional EXPLICIT split list, for splits that do not follow the
            // <name>-0000i-of-0000n.gguf naming scheme. Empty (the default, and every existing
            // call site) keeps the previous behaviour exactly: the split set is derived from
            // LLM_KV_SPLIT_COUNT + llama_get_list_splits(). Added so llama_model_load_from_splits
            // (include/llama.h) has somewhere to put its paths.
            const std::vector<std::string> & splits = {},
            // Optional PARTIAL split set (0-based split numbers) for quantizing individual shards:
            // only split 0 and these splits are opened, the others need not exist on disk, and the
            // tensor-count sanity check is relaxed to "loaded <= declared". nullptr (the default,
            // every existing call site) opens every split exactly as before.
            const std::vector<uint16_t> * only_splits = nullptr,
            // NVFP4_KS activation precision (llama_nvfp4_precision); AUTO = 0 for every existing call site.
            int nvfp4_precision_in = 0);

    // llama_nvfp4_precision chosen for this load (0 AUTO, 1 W4A4, 2 INT8). Read by import_nvfp4().
    int nvfp4_precision = 0;

    // UNIT 1 (S69-PORT-MODEL-INIT-FROM-USER-AND-FILE-PTR-LOADER-2026-09-28-01): metadata-only
    // construction backing llama_model_init_from_user (include/llama.h). No file, no mmap -- tensor
    // BYTES are delivered later by `set_tensor_data` (load_data_for()/load_all_data() call it
    // instead of touching `files`, both gated on it being non-null). `use_mmap` is not a parameter
    // here: mmap needs a file to map and there is none, so it is pinned off, mirroring upstream's
    // own params.use_extra_bufts pin-off for this same entry point (SOURCE-READ: upstream
    // src/llama.cpp:446-457). SOURCE-READ: upstream llama-model-loader.cpp's `else` branch (its ctor,
    // around line 710) for scope -- upstream does NOT populate weights_map from the caller's
    // metadata on this path either (tensor creation there falls back to its buft_list-aware
    // create_tensor()'s F32 default, which THIS fork's simpler create_tensor() -- see loader.h's
    // top-of-file exclusion note -- does not have). CHALLENGE ANSWER (open item, not silently
    // completed): this constructor and the byte-delivery callback wiring are real; end-to-end
    // llm_load_tensors() success through this path additionally needs that F32-fallback ported into
    // this fork's create_tensor(), which this unit does NOT do -- UNMEASURED / next unit's scope.
    llama_model_loader(struct gguf_context * metadata_in,
            llama_model_set_tensor_data_t set_tensor_data_in, void * set_tensor_data_ud_in,
            int ncmoe, bool check_tensors, bool repack_tensors, bool use_thp,
            bool merge_qkv, bool merge_up_gate_exps, bool defer_experts,
            const llama_model_kv_override * param_overrides_p,
            const llama_model_tensor_buft_override * param_tensor_buft_overrides_p);

    // UNIT 1: FILE*-backed construction backing llama_model_load_from_file_ptr. The primary GGUF is
    // read from an already-open FILE* instead of a path -- REAL, not scaffolding: both lower layers
    // already existed in this fork before this unit (SOURCE-READ: llama-mmap.cpp:415
    // `llama_file::llama_file(FILE*)`; ggml/src/gguf.cpp:944 `gguf_init_from_file_ptr`, declared
    // ggml/include/gguf.h:84) and this ctor is mostly wiring them into the same weights_map /
    // create_tensor() path the fname ctor already uses successfully. No split support through this
    // entry point (matches upstream's own scope for llama_model_load_from_file_ptr,
    // src/llama.cpp:488-496: exactly one file, splits always empty).
    llama_model_loader(FILE * file, int ncmoe, bool use_mmap, bool check_tensors, bool repack_tensors,
            bool use_thp, bool merge_qkv, bool merge_up_gate_exps, bool defer_experts,
            const llama_model_kv_override * param_overrides_p,
            const llama_model_tensor_buft_override * param_tensor_buft_overrides_p);

    ~llama_model_loader();

    // UNIT 1 helper, called only by the two new ctors above -- NOT by the pre-existing fname ctor,
    // which keeps its own inline tail exactly as it was (so the byte-identical fname/path load is
    // untouched by this unit). Finishes construction once `meta` and (for the FILE* ctor only)
    // `files`/`weights`/`weights_map` are already populated: n_kv/n_tensors/fver, ftype detection
    // (an empty `weights` for the metadata-only ctor makes this loop a no-op, same as upstream's own
    // externally-supplied-metadata branch), the KV dump log, and the trailing member assignments.
    // Ollama-format blobs whose layout differs from the llama.cpp one (SOURCE-READ ollama llama/compat/llama-ollama-compat.cpp:369-428,
    // compatmigrate/embeddinggemma.go:1-80). Rewrites the in-memory metadata and the by-name tensor index before any tensor is created;
    // the file on disk is never touched. Called by the constructors that read a file, after weights_map is built.
    void apply_ollama_compat();

    // Ollama-format qwen35 blob with an embedded MTP head stored as mtp.* (ours: blk.N.nextn.*). Called by llama_model_load
    // only when MTP is requested, so a load without MTP is unchanged. Returns true when it translated the head. Three norms get
    // +1.0 at read time (compat_norm_add_one); load_all_data gives only those owned storage, so mmap stays on for the rest. LLAMA_OLLAMA_COMPAT=0 is the
    // opt-out (same switch idea as ollama's OLLAMA_LLAMA_CPP_COMPAT=0).
    // RESEARCH-DONE: ollama a8aaf9f llama/compat/llama-ollama-compat.cpp:836-895 + HF Qwen/Qwen3.6-27B norms, see the .cpp.
    bool apply_ollama_qwen35_mtp_compat();

    // F32 tensors (by final name) whose file data gets +1.0 per element at read time, see apply_ollama_qwen35_mtp_compat
    std::vector<std::string> compat_norm_add_one;

    void finish_construction_from_meta(int ncmoe, bool use_mmap, bool check_tensors,
            bool repack_tensors, bool use_thp, bool merge_qkv, bool merge_up_gate_exps,
            bool defer_experts);
    bool import_nvfp4();

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, const bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const enum llm_kv kid, T & result, const bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, const bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, const bool required = true);

    template<typename T>
    bool get_arr(const enum llm_kv kid, T & result, const bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, const bool required = true);

    template<typename T>
    bool get_key(const enum llm_kv kid, T & result, const bool required = true);

    // get array of n <= N_MAX elements, or a single element repeated n times
    // allow_partial: tolerate a stored array SHORTER than n by zero-padding the rest -- e.g. mrope
    // dimension_sections [t,h,w] = 3 vs the expected 4. Ollama qwen3vl/qwen35 GGUFs write 3 sections or omit
    // the key; without this they throw "wrong array length" on load. DEC-S69-MROPE-SECTIONS-TOLERANT.
    // Recovered 2026-09-27 from stranded commit 3e367ef6b (ik_llama_merged) -- it had been lost from HEAD.
    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, const bool required = true,
                        const bool allow_partial = false);

    template<typename T>
    bool get_key_or_arr(const enum llm_kv kid, T & result, uint32_t n, const bool required = true,
                        const bool allow_partial = false);

    bool get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required = true);

    // M-RoPE section split for an arch that ALWAYS applies mrope. Present (3 or 4 entries) -> read, zero-padded.
    // ABSENT -> `fallback` when the arch has a SOURCED default, else throw. Never leave all four at zero:
    // ggml's mrope cache init computes `(i0/2) % sect_dims` (ggml-cpu/ops.cpp) -- zero sections load cleanly
    // and then divide by zero on the first decode. Archs that gate on use_mrope() keep the plain tolerant read.
    void get_mrope_sections(std::array<int, 4> & sections, const std::array<int, 4> * fallback);

    // Sourced defaults (Ollama blobs omit the key). qwen3vl/qwen3vlmoe: HF Qwen3-VL-8B-Instruct and
    // Qwen3-VL-30B-A3B-Instruct config.json text_config.rope_scaling.mrope_section = [24, 20, 20].
    // qwen35/qwen35moe: conversion/qwen.py _QWEN35_DEFAULT_MROPE_SECTION = [11, 11, 10, 0].
    static constexpr std::array<int, 4> MROPE_DEFAULT_QWEN3VL = {24, 20, 20, 0};
    static constexpr std::array<int, 4> MROPE_DEFAULT_QWEN35  = {11, 11, 10, 0};

    const std::string& get_arch_name() const { return arch_name; }

    enum llm_arch get_arch() const {
        if (!arch_resolved) {
            resolved_arch = llm_kv.arch;
            if (resolved_arch == LLM_ARCH_DFLASH &&
                    get_tensor_meta("selector_hidden.weight") != nullptr) {
                resolved_arch = LLM_ARCH_DFLASH2;
            }
            if (resolved_arch == LLM_ARCH_GLM5NEXT_DASHED) {
                // A GGUF whose general.architecture is "glm5-next" matches two enums, GLM5NEXT_DASHED (fork) and GLM5_NEXT (merged
                // upstream, models/glm5-next.cpp); the name scan in llm_arch_from_string returns the lower enum, so the fork graph
                // (build_glm5next.cpp) always won and the upstream builder was unreachable from any file. LLAMA_GLM5NEXT_IMPL=upstream
                // routes the SAME file to the upstream builder, for the parity test (tests/test-glm5next-two-impl-parity.py).
                // Default (unset, or any other value) is unchanged: the fork path.
                resolved_arch = llm_glm5next_impl_resolve(resolved_arch);
            }
            arch_resolved = true;
        }
        return resolved_arch;
    }

    const char * get_tensor_name(int i) const;

    // upstream's O(log n) weights_map lookup -- real improvement over fork's original O(n) linear
    // scan, safe now that weights_map is populated alongside `weights` (see above).
    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight * get_weight(int i) const {
        return get_weight(get_tensor_name(i));
    }

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const char * name) const;
    // adopted from upstream in this merge: additive overload, forwards to the char* form.
    struct ggml_tensor * require_tensor_meta(const std::string & name) const {
        return require_tensor_meta(name.c_str());
    }

    struct ggml_tensor * get_tensor_meta(int i) const {
        return get_tensor_meta(get_tensor_name(i));
    }

    struct ggml_tensor * create_tensor_for(struct ggml_context * ctx, const struct ggml_tensor * cur, bool duplicated);

    // `allow_reshape` (default false = fork's original strict per-dimension check) is upstream's
    // TENSOR_ALLOW_RESHAPE addition: check total element count only. Additive default parameter,
    // so every existing 3-arg call site is unchanged.
    const struct ggml_tensor * check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required, bool allow_reshape = false) const;

    static const int TENSOR_NOT_REQUIRED = 1 << 0;
    static const int TENSOR_DUPLICATED   = 1 << 1;
    static const int TENSOR_SKIP         = 1 << 2;
    // adopted from upstream in this merge (S69-SRC-MODELS-BLOCKED-BY-MISSING-LLAMA-MODEL-BASE-
    // HIERARCHY-2026-09-08-01): the 5th upstream TENSOR_* flag, filling the bit-gap between
    // TENSOR_SKIP (1<<2) and TENSOR_ALLOW_RESHAPE (1<<4) that was this fork's own visible
    // fingerprint of the missing value (survey section C). Upstream ties its real behaviour to a
    // "virtual tensor" concept this fork has not adopted, so this bit is not yet interpreted by
    // create_tensor()/check_tensor_dims() below -- it is a real, correctly-valued constant (same
    // numeric identity upstream gives it), not a stub, but setting it has no effect on tensor
    // creation here until that concept lands. Named as an open item, not silently completed.
    static const int TENSOR_SKIP_IF_VIRTUAL = 1 << 3;
    // adopted from upstream in this merge: additive flag bit; only meaningful once
    // check_tensor_dims is called with allow_reshape=true (create_tensor does this below).
    static const int TENSOR_ALLOW_RESHAPE = 1 << 4;
    // upstream: read rows on demand (lazy PLE / engram tables). Upstream model classes pass it; this fork's
    // create_tensor tests only the bits it knows, so such a tensor is loaded in full (row
    // S69-ENGREV-UPSTREAM-MERGE-LOADER-LAZY-READ-ASYNC-UPLOAD-PORT-2026-11-04-01).
    static const int TENSOR_READ_LAZY = 1 << 5;

    struct ggml_tensor * create_tensor(struct ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne, int flags = 0);

    // Target model a draft head borrows token_embd / output_norm / output from (llama_model_params::model_shared).
    // Consulted only by borrow_shared_tensor(); null for every ordinary load.
    const struct llama_model * model_shared = nullptr;

    // A draft head that shares the target's embeddings carries none of token_embd, output_norm or output;
    // take them from the target. Returns null unless the file is missing token_embd, so a model that ships
    // its own tensors is never affected. Throws on a missing target or a shape disagreement.
    struct ggml_tensor * borrow_shared_tensor(const std::string & name, const std::vector<int64_t> & ne, int flags);

    struct ggml_tensor * create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base,
            const std::string & name, const std::vector<int64_t> & ne, size_t offset, bool required = true);

    // `partial` (default false, adopted from upstream in this merge) allows n_created <
    // n_tensors when the caller opts in; default preserves the exact-match behavior every
    // existing 0-arg call site (src/llama.cpp) already relies on.
    void done_getting_tensors(bool partial = false) const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr, bool use_thp = false);

    void build_expert_tensor_index(const llama_hparams & hparams);

    bool should_defer_expert_mmaps() const;

    bool has_anonymous_mapping() const;

    void drop_mmap_expert_pages() const;

    void build_ple_tensor_index();

    // Fold tensors flagged TENSOR_READ_LAZY into ple_tensor_index. build_ple_tensor_index() finds the tables by NAME before the
    // arch runs; this honours the flag itself so an arch that marks another tensor lazy is not silently loaded in full. Call after
    // create_tensors(). Tensors the name index already covers are skipped (no double count).
    void index_lazy_flagged_tensors();

    bool should_defer_ple_mmaps() const;

    void apply_ple_mmap_policy() const;

    // File holds deferred ranges (PLE, engram): skip prefetch and pinning for it.
    bool file_has_deferred_ple(int idx) const;

    bool ple_range_overlaps(int idx, size_t first, size_t last) const;

    // deferred bytes of the index inside [first, last) of file idx
    size_t ple_deferred_bytes_in(int idx, size_t first, size_t last) const;

    // byte range within a mapped file
    struct mapping_range {
        size_t first; // inclusive
        size_t last;  // exclusive
    };

    // sorted, non-overlapping ranges of file 'idx' that back the tensors of 'ctx'
    // ranges owned by other contexts are excluded, so that a buffer wrapping them won't pin
    // data that this context never reads (lazy tensors, -cmoe tensors, ...)
    // (upstream's get_mapping_ranges(idx, ctx) is not declared: its body needs ctx_map, which this loader has no
    //  member for, and its only caller is upstream's model load path)

    // one span [first, last) over the tensors of 'ctx' in file 'idx' (restored 2026-11-05; tests/test-loader-mapping-ranges.cpp
    // compares it against get_mapping_ranges below)
    void get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const;

    // ADOPTED from unslothai/llama.cpp PR #152. Near-adjacent runs are merged, so a normal
    // layout yields exactly one range and only a large foreign tensor splits it.
    void get_mapping_ranges(std::vector<std::pair<size_t, size_t>> & ranges, void ** addr, int idx, ggml_context * ctx) const;

    // read a byte range of a weight's data
    // with mmap, returns a pointer into the mapping, otherwise reads into buf and returns buf
    const void * load_data_range(const llama_tensor_weight & w, size_t offs, size_t size, void * buf) const;

    // for backwards compatibility, does not support ggml-backend (llama-quantize.cpp reads tensors through it)
    void load_data_for(struct ggml_tensor * cur) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            struct llama_model * model,
            llama_buf_map & bufs_mmap,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    // adopted from upstream in this merge: real, working method -- releases ONE weight's mmap
    // pages by reusing the SAME mechanism drop_mmap_expert_pages() already uses
    // (llama_mmap::dontneed_fragment over [offs, offs+nbytes) on the file's own mapping).
    // Confirmed live: src/llama-quant.cpp does `if (ml.use_mmap) { ml.unmap_weight(weight); }`.
    void unmap_weight(const llama_tensor_weight & w) const;

    // adopted from upstream in this merge: trivial, safe wrapper around the free function below.
    // Not currently called by a resolved consumer, but zero-risk and kept for parity with
    // include/llama.h's already-public llama_ftype_name.
    std::string ftype_name() const;

    // adopted from upstream in this merge: trivial diagnostic, uses only already-existing fields.
    void print_info() const;
};

void llm_load_arch(llama_model_loader & ml, llama_model & model);

void llm_load_hparams(llama_model_loader & ml, llama_model & model, bool ignore_vocab = false);

// sliding_window_pattern in either form it is published in (llama.cpp #29042, MIT): an array of one
// flag per layer (what llama-model-saver writes, nextn layers included) is copied into
// hparams.is_swa_impl as-is; a scalar is a period handed to set_swa_pattern(period, dense_first).
// Returns false and leaves hparams untouched when the key is absent. Defined in llama-hparams.cpp.
bool llm_read_swa_pattern(llama_model_loader & ml, llama_hparams & hparams, bool dense_first = false);

// As llm_read_swa_pattern, but an absent key applies set_swa_pattern(n_pattern, dense_first) --
// the arch's default period, exactly what the period-only loaders did before.
void llm_load_swa_pattern(llama_model_loader & ml, llama_hparams & hparams, uint32_t n_pattern, bool dense_first = false);

// Confirmed live: src/llama.cpp's llama_model_load() calls
// create_tensors_helper_interface::instance(ml, model), and the concrete implementation
// (create_tensors_helper, in src/llama-load-tensors.cpp -- 6367 lines, already resolved, not
// part of this conflict cluster) uses the SIMPLE create_tensor(ctx, name, ne, flags) above, not
// upstream's buft_list-driven overload. See loader.cpp's top-of-file exclusion note for the full
// chain of evidence (llama_model_load_from_splits/llama_model_init_from_user, upstream's other
// public entry points into a buft_list-aware loading flow, have NO definition anywhere in src/).

// Prism ternary Hadamard rotations (ik_llama.cpp #2550, MIT); no-op unless the GGUF declares them
void llm_load_hadamard(llama_model_loader & ml, llama_model & model);

struct create_tensors_helper_interface {
    virtual ~create_tensors_helper_interface() = default;
    virtual bool create_tensors() = 0;
    virtual std::map<ggml_backend_buffer_type_t, ggml_context *> & get_ctx_map() = 0;
    virtual size_t get_ctx_size() const = 0;

    static std::unique_ptr<create_tensors_helper_interface> instance(llama_model_loader & ml, llama_model & model);
};
