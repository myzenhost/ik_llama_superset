#pragma once

#include "ggml.h"

struct ggml_graph_node_properties {
    void * node_address;
    ggml_op   node_op;
    ggml_type type;
    void * view_src;
    size_t view_offs;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void * src_address[GGML_MAX_SRC];
    // A source's layout, not just its address: a node whose own shape and source addresses are
    // unchanged can still read its sources through different strides (a permuted view of the same
    // buffer), and the captured kernels bake the old strides in. Upstream compares these too.
    int64_t src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t  src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    // Types too: F16 and BF16 (or any two equal-size types) give identical ne/nb/addresses, and the captured graph
    // bakes in the kernel chosen for the OLD type. MEASURED 2026-09-28: LIGHTNING_INDEXER type_K=bf16 right after
    // the same-shape f16 case replayed the f16 WMMA kernel on bf16 data (ERR ~0.5, 3/3), and passed 144/144 with
    // GGML_CUDA_DISABLE_GRAPHS=1. Upstream memcmps the whole ggml_tensor, which includes the type; the fork's own
    // CANN backend stores node_type / src_type (ggml-cann/common.h:220,226).
    ggml_type node_type;
    ggml_type src_type[GGML_MAX_SRC];
};

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph() {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
        // Graphs are evicted by the context's sweep now (common.cuh cuda_graph()); without this the
        // device array of copy destinations leaked with every evicted graph.
        // Runtime already unloaded (a context torn down at process exit) is not an error here: the
        // array never used to be freed at all.
        if (dest_ptrs_d != nullptr) {
            const cudaError_t err = cudaFree(dest_ptrs_d);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
                CUDA_CHECK(err);
            }
        }
    }
    uint64_t uid = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> params;
    bool disable_due_to_gpu_arch = false;
    // Kept for GGML_CUDA_GRAPH_LEGACY=1 only (the pre-warmup policy, for A/B): 4 consecutive
    // property changes disabled this graph for the life of the context.
    bool disable_due_to_too_many_updates = false;
    bool disable_due_to_failed_graph_capture = false;
    int number_consecutive_updates = 0;
    // Upstream's policy (llama.cpp ggml-cuda.cu ggml_backend_cuda_graph_compute, SOURCE-READ of
    // master fetched 2026-10-18, sha256 68f09269...7777, lines 4800-4826): a graph is captured
    // only after two consecutive calls with unchanged properties; a later change resets this and
    // the node runs directly until it is stable again. Nothing is ever disabled for good.
    bool warmup_complete = false;
    std::vector<ggml_graph_node_properties> ggml_graph_properties;
    bool use_cpy_indirection = false;
    std::vector<char *> cpy_dest_ptrs;
    char ** dest_ptrs_d = nullptr;
    int dest_ptrs_size = 0;
    // Index to allow each cpy kernel to be aware of it's position within the graph
    // relative to other cpy nodes.
    int graph_cpynode_index = -1;

    // Last use, in ggml_time_us() microseconds. Required by the multi-graph cache in
    // ggml_backend_cuda_context (common.cuh): that context keeps a map from first-node pointer to
    // graph -- so a computation split across CPU/GPU, e.g. --n-cpu-moe, can hold more than one --
    // and sweeps it every 5s, evicting graphs unused for >=10s. Without this field that eviction
    // code does not compile; the merge brought the cache over and left the field behind.
    // Initialised to 0, which the sweep reads as "very old". That is harmless: a graph is stamped
    // with the current time on the same call that creates it, before any sweep can see it.
    int64_t last_used_time = 0;
#endif
};

