#pragma once

#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

// tensor-name regex that keeps the FFN tensors of layer il (the part of it NOT named by lf) in host memory. The returned pointer
// stays valid for the life of the process: the fit stores it in the caller's tensor_buft_overrides buffer.
const char * common_fit_overflow_pattern(size_t il, common_layer_fraction_t lf);

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
struct common_fit_extra_model {
    const char * path_model;
    llama_model_params * mparams;
    llama_context_params * cparams;
    bool shares_model;
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
      const common_fit_extra_model * extra,                  // model to fit alongside the main one, nullptr if there is none
                     ggml_log_level   log_level,             // minimum log level to print during fitting, lower levels go to debug log
                               bool   moe_cache_auto = false); // set cparams->moe_cache_size from the free memory (one GPU, MoE model)

// --moe-cache-mib auto: how big the GPU expert cache is when EVERY routed expert stays in host memory. Returns bytes, 0 = use no cache,
// with the reason in *why. Pure, so it is unit-tested without a GPU (tests/test-fit-moe-cache-auto.cpp).
//   target            device memory the fit may use (free - margin)
//   mem_cpu_moe       device memory use with all MoE tensors in host memory (the fork's dry-run context does NOT allocate the cache,
//                     so this is also the use with the cache excluded)
//   mem_model_full / mem_model_cpu_moe   model part of the device use with the experts on the device / in host memory
//   cache_min_size    llama_model_moe_cache_min_size: smallest budget that caches one token's experts in every layer
int64_t common_fit_moe_cache_pick(int64_t target, int64_t mem_cpu_moe, int64_t mem_model_full, int64_t mem_model_cpu_moe,
                                  size_t cache_min_size, std::string * why);

// per-layer share of the projected model memory that the multi-device context reduction subtracts per device. Pure.
// Returns 0 when min(n_gpu_layers, hp_ngl) is 0 (the user passed -ngl 0): that divisor used to be 0.
// Row S69-ENGREV-ACCELERATE-FIT-NGL0-DIV-BY-ZERO-MULTIGPU-2026-10-27-01 (tests/test-fit-ngl0.cpp).
int64_t common_fit_model_per_layer(int64_t sum_projected_model, int32_t n_gpu_layers, uint32_t hp_ngl);

// one device of the post-fit receipt, bytes
struct common_fit_receipt_dev {
    std::string name;
    int64_t     used;   // projected use with the final parameters
    int64_t     free;   // free before the load
    int64_t     margin; // margin the fit was asked to leave
};

// The block common_fit_params logs (at INFO, WARN when host_warn) after a successful fit: chosen n_gpu_layers, context before (0 = auto)
// and after, per-device projected use / free / margin, host-resident bytes versus free host memory. *host_warn is set when
// host_resident > host_free - host_margin (mmap paging thrash predicted); that block then carries a "fit: WARN" line. Pure.
// Row S69-ENGREV-ACCELERATE-FIT-RESULT-RECEIPT-2026-10-27-01 (tests/test-fit-receipt.cpp).
std::string common_fit_receipt(int32_t n_gpu_layers, uint32_t n_ctx_before, uint32_t n_ctx_after,
                               const std::vector<common_fit_receipt_dev> & devs,
                               int64_t host_resident, int64_t host_free, int64_t host_margin, bool * host_warn);

// one device of the post-warmup check: what the fit projected versus what the backend actually allocated, bytes
struct common_fit_measured_dev {
    std::string name;
    int64_t     projected;
    int64_t     measured;
};

// Text of the post-warmup "measured next to projected" block. *warn is set when any device has measured use more than 5 percent
// above its projection (the fit under-counted, so a later request can OOM); that block then carries a "fit: WARN" line per
// device. Empty string for an empty list. Pure. Row S69-FIT-PROJECTED-VS-MEASURED-PEAK-NANO-VLLM-PICK-2026-09-27-01
// (absorbed by S69-ENGREV-ACCELERATE-FIT-RESULT-RECEIPT-2026-10-27-01; tests/test-fit-receipt.cpp).
std::string common_fit_measured_report(const std::vector<common_fit_measured_dev> & devs, bool * warn);

// Called by common_init_from_params after the warmup decode: reads the per-device bytes the context really holds
// (llama_get_memory_breakdown) and logs them beside the projection common_fit_params remembered (INFO, WARN when over 5 percent).
// Silent when no fit ran in this process. Never throws.
void common_fit_log_measured(const struct llama_context * ctx);

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

// one line of the per-buffer-type memory breakdown, in bytes: a device, the host, or any other buffer type
struct common_memory_breakdown_row {
    enum kind_t { DEVICE, HOST, BUFFER } kind = BUFFER;
    std::string name;
    std::string description;   // device only (vendor prefix stripped)
    int64_t total       = -1;  // device only: -1 = not applicable
    int64_t free        = -1;  // device only
    size_t  model       = 0;
    size_t  context     = 0;
    size_t  compute     = 0;
    int64_t unaccounted = 0;   // device only: total - free - self (memory used by something else)

    size_t self() const { return model + context + compute; }
};

// what common_memory_breakdown_print prints, as data: devices first, then the host, then any other buffer type
std::vector<common_memory_breakdown_row> common_memory_breakdown_collect(const llama_context * ctx);

void common_memory_breakdown_print(const llama_context * ctx);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);

// bytes the context's buffers hold (model + KV cache + compute), split by where they live: host memory vs a device (GPU) buffer.
// Read from llama_get_memory_breakdown(); the numbers are what the backend allocated, not an estimate.
// Feeds the Ollama API's /api/ps size / size_vram (ollama server/routes.go:2712-2760, sched.go loadedModels).
void common_memory_totals(const struct llama_context * ctx, uint64_t & host_bytes, uint64_t & device_bytes);
