#pragma once

// Per-layer K/V cache types (-ctk-first/-ctk-last/-ctv-first/-ctv-last) for the memory-module KV cache.
//
// The legacy cache (llama_kv_cache_init, src/llama.cpp) has always honoured these. llama_kv_cache -- the
// memory-module cache that every dispatched src/models/ class decodes from -- created every layer with one
// type_k/type_v, so on those models the flags were accepted and silently did nothing (MEASURED 2026-10-17:
// olmo2 with -ctv turbo2 -ctv-first q8_0,2 -ctv-last q8_0,2 allocated turbo2 V on all 32 layers).
//
// The memory module is built through ~20 create_memory() arms and several composite caches (iSWA, hybrid,
// DSA, MSA), each forwarding type_k/type_v. Rather than widen every one of those signatures, llama_context
// publishes the resolved per-layer policy for the duration of create_memory() (llama_kv_layer_types_scope)
// and the one place that allocates layer tensors, llama_kv_cache's constructor, reads it.

#include "ggml.h"

#include <cstdint>
#include <string>

struct llama_kv_layer_types {
    ggml_type type_k_first;
    ggml_type type_k_last;
    ggml_type type_v_first;
    ggml_type type_v_last;
    int32_t   n_k_first;
    int32_t   n_k_last;
    int32_t   n_v_first;
    int32_t   n_v_last;
    int64_t   n_layer;   // the count "last N" is measured from; same value the legacy cache uses
};

// Same rule as the legacy cache (src/llama.cpp, llama_kv_cache_init): an override applies when its type differs
// from the base type and its count is positive; "last" wins over "first" where both cover a layer.
static inline void llama_kv_layer_types_apply(const llama_kv_layer_types & p, int64_t il,
                                              ggml_type & type_k, ggml_type & type_v) {
    const ggml_type base_k = type_k;
    const ggml_type base_v = type_v;
    if (p.type_k_first != base_k && p.n_k_first > 0 && il < p.n_k_first)            type_k = p.type_k_first;
    if (p.type_k_last  != base_k && p.n_k_last  > 0 && il >= p.n_layer - p.n_k_last) type_k = p.type_k_last;
    if (p.type_v_first != base_v && p.n_v_first > 0 && il < p.n_v_first)            type_v = p.type_v_first;
    if (p.type_v_last  != base_v && p.n_v_last  > 0 && il >= p.n_layer - p.n_v_last) type_v = p.type_v_last;
}

// Coverage check (row S69-ENGREV-APHRODITE-KVLAYER-COVERAGE-WARN-2026-10-27-01). An override can be requested (type differs from the base
// type, count > 0) and yet reach no layer of a given cache: first/last are ABSOLUTE layer indices over n_layer, so on a hybrid model whose
// first N layers are recurrent, -ctk-first N changes nothing in the attention cache and used to say nothing (accepted and silently ignored).
// Returns "" when every requested override took effect on at least one layer of this cache; otherwise one WARN text naming each flag
// and its requested count. n_k_changed / n_v_changed = layers of THIS cache whose K / V type the policy changed.
static inline std::string llama_kv_layer_types_uncovered(const llama_kv_layer_types & p, ggml_type base_k, ggml_type base_v,
                                                         int64_t n_k_changed, int64_t n_v_changed, bool has_v) {
    std::string out;
    auto add = [&](const char * flag, int32_t n) {
        out += std::string(out.empty() ? "" : ", ") + flag + " " + std::to_string(n);
    };
    if (n_k_changed == 0) {
        if (p.type_k_first != base_k && p.n_k_first > 0) add("-ctk-first", p.n_k_first);
        if (p.type_k_last  != base_k && p.n_k_last  > 0) add("-ctk-last",  p.n_k_last);
    }
    if (has_v && n_v_changed == 0) {
        if (p.type_v_first != base_v && p.n_v_first > 0) add("-ctv-first", p.n_v_first);
        if (p.type_v_last  != base_v && p.n_v_last  > 0) add("-ctv-last",  p.n_v_last);
    }
    if (out.empty()) {
        return out;
    }
    return "per-layer KV type override requested (" + out + ") but it reaches no layer of this cache: first/last count absolute layer "
           "indices over " + std::to_string(p.n_layer) + " layers, and this cache holds only some of them (recurrent/filtered layers have no KV); "
           "the override is ignored here";
}

// The policy in force for the current thread, or nullptr. Defined in llama-kv-cache.cpp.
const llama_kv_layer_types * llama_kv_layer_types_current();

class llama_kv_layer_types_scope {
public:
    explicit llama_kv_layer_types_scope(const llama_kv_layer_types * p);
    ~llama_kv_layer_types_scope();
    llama_kv_layer_types_scope(const llama_kv_layer_types_scope &) = delete;
    llama_kv_layer_types_scope & operator=(const llama_kv_layer_types_scope &) = delete;
private:
    const llama_kv_layer_types * prev;
};
