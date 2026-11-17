// ggml_gallocr_alloc_graph() on a graph that has the same node count, leaf count and tensor sizes as the previous one, but whose leaves appear in a
// different ORDER (a weight where the previous graph had an input, and vice versa). Before the leaf check in ggml_gallocr_needs_realloc() the allocator
// re-initialised leaf i from the previous graph's allocation for leaf i and indexed bufts[-1] for the input that sat at a weight's index.
// This is what per-sequence LoRA graphs do when the adapter that is applied first changes (docs/agent-reports/wire-serving-features.md).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

int main() {
    ggml_backend_t be = ggml_backend_cpu_init();
    // two weights: allocated up front in a backend buffer, never by the graph allocator
    ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * wa = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 8);
    ggml_tensor * wb = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 8);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    std::vector<float> av(8), bv(8, 2.0f);
    for (int i = 0; i < 8; ++i) av[i] = 1.0f + i;
    ggml_backend_tensor_set(wa, av.data(), 0, sizeof(float) * 8);
    ggml_backend_tensor_set(wb, bv.data(), 0, sizeof(float) * 8);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));

    // x and m are inputs (no data: the graph allocator places them). r = t1 + t4 with t4 = ((first * x) * m) * wb. The two graphs have the same ops
    // with the same kinds (weight / input / node) at the same source positions, the same node count and the same leaf count -- only WHERE the weights
    // are first reached differs, which moves the leaves:
    //   graph 1 (t1 = wb * x):  leaves [wb, x, wa, m]
    //   graph 2 (t1 = wa * x):  leaves [wa, x, m, wb]     <- leaf 2 is an input here and a weight there
    auto build = [&](bool first_is_wb, ggml_context ** out_ctx, ggml_tensor ** x, ggml_tensor ** m, ggml_tensor ** out) {
        ggml_init_params p = { ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true };
        ggml_context * c = ggml_init(p);
        *x = ggml_new_tensor_1d(c, GGML_TYPE_F32, 8);
        *m = ggml_new_tensor_1d(c, GGML_TYPE_F32, 8);
        ggml_set_input(*x);
        ggml_set_input(*m);
        ggml_tensor * t1 = ggml_mul(c, first_is_wb ? wb : wa, *x);
        ggml_tensor * t2 = ggml_mul(c, wa, *x);
        ggml_tensor * t3 = ggml_mul(c, t2, *m);
        ggml_tensor * t4 = ggml_mul(c, t3, wb);
        *out = ggml_add(c, t1, t4);
        ggml_set_output(*out);
        *out_ctx = c;
        ggml_cgraph * g = ggml_new_graph_custom(c, 32, false);
        ggml_build_forward_expand(g, *out);
        return g;
    };

    ggml_context *c1, *c2; ggml_tensor *x1, *m1, *o1, *x2, *m2, *o2;
    ggml_cgraph * g1 = build(true,  &c1, &x1, &m1, &o1);
    ggml_cgraph * g2 = build(false, &c2, &x2, &m2, &o2);
    CHECK(g1->n_nodes == g2->n_nodes && g1->n_leafs == g2->n_leafs);
    printf("graph 1: %d nodes %d leafs, leaf 2 is %s; graph 2: %d nodes %d leafs, leaf 2 is %s\n", g1->n_nodes, g1->n_leafs,
           (g1->leafs[2] == wa || g1->leafs[2] == wb) ? "a weight" : "an input", g2->n_nodes, g2->n_leafs,
           (g2->leafs[2] == wa || g2->leafs[2] == wb) ? "a weight" : "an input");
    CHECK((g1->leafs[2] == wa || g1->leafs[2] == wb) != (g2->leafs[2] == wa || g2->leafs[2] == wb));

    std::vector<float> xv(8, 2.0f), mv(8, 3.0f), res(8);
    auto expect = [&](int i, bool first_is_wb) { // t1 + ((wa*x)*m)*wb
        const float a = av[i], b = bv[i], x = 2.0f, m = 3.0f;
        return (first_is_wb ? b * x : a * x) + ((a * x) * m) * b;
    };
    CHECK(ggml_gallocr_alloc_graph(galloc, g1));
    ggml_backend_tensor_set(x1, xv.data(), 0, sizeof(float) * 8);
    ggml_backend_tensor_set(m1, mv.data(), 0, sizeof(float) * 8);
    ggml_backend_graph_compute(be, g1);
    ggml_backend_tensor_get(o1, res.data(), 0, sizeof(float) * 8);
    CHECK(res[0] == expect(0, true) && res[7] == expect(7, true));

    CHECK(ggml_gallocr_alloc_graph(galloc, g2));   // crashed here before the leaf check
    ggml_backend_tensor_set(x2, xv.data(), 0, sizeof(float) * 8);
    ggml_backend_tensor_set(m2, mv.data(), 0, sizeof(float) * 8);
    ggml_backend_graph_compute(be, g2);
    ggml_backend_tensor_get(o2, res.data(), 0, sizeof(float) * 8);
    CHECK(res[0] == expect(0, false) && res[7] == expect(7, false));
    printf("graph 2 allocated and computed: out[0]=%g (expect %g) out[7]=%g (expect %g)\n", res[0], expect(0, false), res[7], expect(7, false));

    ggml_gallocr_free(galloc);
    ggml_free(c1); ggml_free(c2);
    ggml_backend_buffer_free(wbuf); ggml_free(wctx);
    ggml_backend_free(be);
    if (n_fail) { fprintf(stderr, "test-gallocr-leaf-order: %d failure(s)\n", n_fail); return 1; }
    printf("test-gallocr-leaf-order: all checks passed\n");
    return 0;
}
