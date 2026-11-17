#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

struct llama_model;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer_type;

// Keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer (ik_llama #2644 base, merged
// 2026-11-03 with upstream llama.cpp #30112/#30637: SLRU policy, large-batch copy_experts, memory_breakdown). Each layer has
// a slot map in host memory; when the scheduler copies it to the device, the copy callback uploads the missing experts.
class llama_moe_cache {
public:
    // backends are all the backends of the context; each GPU gets its own cache for the layers assigned to it
    llama_moe_cache(const llama_model & model,
            const std::vector<ggml_backend *> & backends,
            const std::vector<ggml_backend_buffer_type *> & bufts,
            size_t size,
            int policy = 0,        // enum llama_moe_cache_policy; the CPU share selects classlru
            float cpu_share = 0.0f); // share of the misses computed on the CPU, 0 = off, < 0 = measure at load
    ~llama_moe_cache();

    // the device that caches layer il, nullptr if the layer is not cached
    ggml_backend * backend(int32_t il) const;

    // the slot map of layer il if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses left to the CPU in the CPU map of the layer (get_cpu_map) and looks up the slot map of that graph
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part = false) const;

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others;
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const;

    // the experts of tensor in the cache, nullptr if tensor is not cached
    ggml_tensor * get_experts(const ggml_tensor * tensor) const;

    // ggml_backend_sched copy callback; returns false if src is not a slot map (QUERY: would handle it; EXECUTE: did it)
    bool copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
            ggml_backend_sched_copy_phase phase);

    // large batches: copy the experts of w that are in the cache, from expert e up to expert last, to the copy dst of w.
    // Returns the number of experts copied, 0 if expert e is not cached. Reads only: the generation experts stay cached.
    int64_t copy_experts(ggml_backend * backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    // device and host bytes held by the cache, per buffer type
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
