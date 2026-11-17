// Two-way test for the opt-in --cpu-flush-denormals switch (ggml_cpu_set_flush_denormals).
//   flag OFF: 1e-40f * 1.0f stays a non-zero denormal (default path unchanged)
//   flag ON : the same graph returns exactly 0 (FTZ/DAZ in every compute worker)
//   flag OFF again: non-zero again, proving the calling thread's control word is restored on exit
// Run with 1 and 2 threads so both the caller-thread and the secondary-worker entry are covered.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>

static int count_nonzero(int n_threads, bool flush) {
    ggml_cpu_set_flush_denormals(flush);

    struct ggml_init_params ip = { /*.mem_size =*/ 8u * 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    struct ggml_context * ctx = ggml_init(ip);
    const int n = 4096;
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    float * pa = (float *) a->data;
    float * pb = (float *) b->data;
    for (int i = 0; i < n; i++) {
        pa[i] = 1e-40f; // denormal (FLT_MIN is ~1.18e-38)
        pb[i] = 1.0f;
    }
    struct ggml_tensor * c = ggml_mul(ctx, a, b);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    int nz = 0;
    const float * pc = (const float *) c->data;
    for (int i = 0; i < n; i++) {
        nz += pc[i] != 0.0f;
    }
    ggml_free(ctx);
    return nz;
}

int main() {
    int fails = 0;
    const int n = 4096;
    for (int nt = 1; nt <= 2; nt++) {
        const int off1 = count_nonzero(nt, false);
        const int on   = count_nonzero(nt, true);
        const int off2 = count_nonzero(nt, false);
        printf("threads=%d  off=%d  on=%d  off-after=%d  (of %d)\n", nt, off1, on, off2, n);
        if (off1 != n) { printf("FAIL: flag off must keep the denormal\n"); fails++; }
        if (on   != 0) { printf("FAIL: flag on must flush to zero\n"); fails++; }
        if (off2 != n) { printf("FAIL: control word not restored after the flag-on run\n"); fails++; }
    }
    ggml_cpu_set_flush_denormals(false);
    printf(fails ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return fails ? 1 : 0;
}
