// Unit test for ggml/src/ggml-cuda/graph-key.h (CUDA-graph cache key), CPU only.
// Row S69-ENGREV-THECODACUS-CUDA-GRAPH-KEY-EMPTY-GUARD-2026-10-27-01.
// An empty cgraph (n_nodes == 0, e.g. an all-view split) must give the fixed key nullptr without reading
// nodes[0] or nodes[-1]. The test hands the key function a NULL nodes array: a read of nodes[0] crashes.
#include "../ggml/src/ggml-cuda/graph-key.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

int main() {
    // 1. empty graph, both policies, nodes == NULL and also a non-NULL array with n_nodes == 0
    {
        ggml_cgraph g = {};
        g.n_nodes = 0;
        g.nodes   = nullptr;
        CHECK(ggml_cuda_graph_key_of(&g, false) == nullptr);
        CHECK(ggml_cuda_graph_key_of(&g, true)  == nullptr);
        ggml_tensor * dummy[1] = { nullptr };
        g.nodes = dummy;
        CHECK(ggml_cuda_graph_key_of(&g, false) == nullptr);
        CHECK(ggml_cuda_graph_key_of(&g, true)  == nullptr);
        g.n_nodes = -1;
        CHECK(ggml_cuda_graph_key_of(&g, false) == nullptr);
    }
    // 2. non-empty graphs keep their behaviour: legacy key is nodes[0]; default key depends on n_nodes and shapes
    {
        ggml_tensor a = {}; ggml_tensor b = {}; ggml_tensor c = {};
        a.ne[0] = 8; a.ne[1] = 1; b.ne[0] = 8; b.ne[1] = 1; c.ne[0] = 8; c.ne[1] = 4;
        ggml_tensor * n2[2] = { &a, &b };
        ggml_tensor * n3[3] = { &a, &b, &c };
        ggml_cgraph g2 = {}; g2.n_nodes = 2; g2.nodes = n2;
        ggml_cgraph g3 = {}; g3.n_nodes = 3; g3.nodes = n3;
        CHECK(ggml_cuda_graph_key_of(&g2, true) == (const void *) &a);
        CHECK(ggml_cuda_graph_key_of(&g3, true) == (const void *) &a);
        CHECK(ggml_cuda_graph_key_of(&g2, false) != nullptr);
        CHECK(ggml_cuda_graph_key_of(&g2, false) != ggml_cuda_graph_key_of(&g3, false));
        CHECK(ggml_cuda_graph_key_of(&g2, false) == ggml_cuda_graph_key_of(&g2, false));
    }
    if (g_fail) { std::printf("test-cuda-graph-key-empty: %d FAILURES\n", g_fail); return 1; }
    std::printf("test-cuda-graph-key-empty: OK\n");
    return 0;
}
