#pragma once
// CUDA-graph cache key for a ggml_cgraph (host-only, no CUDA types, so a CPU test can compile it).
//
// Row S69-ENGREV-THECODACUS-CUDA-GRAPH-KEY-EMPTY-GUARD-2026-10-27-01. SOURCE-READ: thecodacus-llama.cpp
// commit 4880a0e2c (MIT) adds `if (cgraph->n_nodes <= 0) return 0;` before its shape-mixed key reads
// nodes[0] / nodes[n_nodes-1]. A graph split that holds only views can reach graph compute with
// n_nodes == 0; without the guard the key read nodes[-1] and dereferenced garbage.
#include "../ggml-impl.h" // struct ggml_cgraph
#include <cstdint>

// An empty graph has nothing to capture: callers get the fixed key nullptr (one shared, never-replayed entry).
static inline const void * ggml_cuda_graph_key_of(const ggml_cgraph * cgraph, bool legacy) {
    if (cgraph->n_nodes <= 0) {
        return nullptr;
    }
    if (legacy) {
        return cgraph->nodes[0];
    }
    // nodes[0] alone (upstream's key too) is the same pointer for every graph built from one
    // compute-meta buffer, so a speculative verify batch (n=2..4), the n=1 decode and an MTP draft
    // graph all landed on ONE cache entry and every call looked like a property change. Mixing in
    // the node count and the token dimensions of the first and last node gives each recurring shape
    // its own entry. A hash collision is harmless: the full property comparison still decides
    // whether a captured graph may be replayed, so it costs a re-capture, never a wrong result.
    const ggml_tensor * first = cgraph->nodes[0];
    const ggml_tensor * last  = cgraph->nodes[cgraph->n_nodes - 1];
    uint64_t h = (uint64_t)(uintptr_t) first;
    auto mix = [&h](uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
    mix((uint64_t) cgraph->n_nodes);
    mix((uint64_t) first->ne[0]);
    mix((uint64_t) first->ne[1]);
    mix((uint64_t) last->ne[0]);
    mix((uint64_t) last->ne[1]);
    return (const void *)(uintptr_t) h;
}
