// Counts how many ops of the parakeet local-attention pad chain the scheduler hands to the
// CPU instead of CUDA.
//
// Why a reconstruction and not the real model: there is no parakeet checkpoint on this host
// (a sweep of 654,957 files across three roots found zero .nemo files, with a voxtral control
// that did return 14/14 hits), so the encoder graph cannot be produced by loading a model.
// What is reproduced here are the SEVEN ggml_pad_ext() call sites of the local-attention
// branch in examples/mtmd/models/parakeet.cpp, at their real shapes and with their real pad
// arguments, including the two that carry a non-zero LOW pad (K_cur and V_cur, lp1 =
// att_left = 128). Each pad feeds a ggml_scale, an op CUDA always supports, so a pad that
// lands on the CPU produces a genuine backend boundary rather than a lone island.
//
// The measurement is the same in both directions: build the graph, schedule it over
// {CUDA0, CPU}, then ask ggml_backend_sched_get_tensor_backend() which backend each node was
// assigned to. Run it against a ggml.dll built before the kernel change and after it.
//
// usage: test-pad-sched-parakeet [n_layer]     (default 24, the parakeet-tdt-0.6b encoder)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h" // ggml_backend_cpu_init (ggml-cpu.h:152); MinGW reached it transitively, MSVC does not
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// examples/mtmd/models/parakeet.cpp:3-4
static const int PARAKEET_LOCAL_ATTN_THRESHOLD = 8192;
static const int PARAKEET_LOCAL_ATTN_WINDOW    = 128;

int main(int argc, char ** argv) {
    const int n_layer = argc > 1 ? atoi(argv[1]) : 24;

    // parakeet-tdt-0.6b encoder geometry
    const int n_state = 1024;
    const int n_head  = 8;
    const int d_head  = n_state / n_head;   // 128

    // n_time must exceed the threshold or parakeet does not take the local-attention branch
    // at all, and these pads never appear. Deliberately not a multiple of chunk, so the
    // need_padding sites are exercised too.
    const int n_time = 8300;

    const int  att_left      = PARAKEET_LOCAL_ATTN_WINDOW;
    const int  att_right     = PARAKEET_LOCAL_ATTN_WINDOW;
    const int  window_size   = att_left + att_right + 1;
    const int  chunk         = att_left + att_right;
    const int  n_group       = (n_time + chunk - 1) / chunk;
    const int  n_time_padded = n_group * chunk;
    const int  n_kv_chunk    = chunk + window_size - 1;
    const int  n_kv_dense    = n_kv_chunk * n_group;
    const bool need_padding  = n_time_padded > n_time;

    if (n_time <= PARAKEET_LOCAL_ATTN_THRESHOLD) {
        fprintf(stderr, "n_time %d does not trigger local attention\n", n_time);
        return 2;
    }

    ggml_backend_t backend_cuda = ggml_backend_cuda_init(0, NULL, NULL);
    if (backend_cuda == NULL) {
        fprintf(stderr, "FATAL: no CUDA backend; this test measures CUDA-vs-CPU assignment\n");
        return 2;
    }
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();

    ggml_backend_t backends[] = { backend_cuda, backend_cpu };
    // op_offload true: the scheduler default every in-tree caller passes (ggml-backend.h:372 added the parameter)
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, GGML_DEFAULT_GRAPH_SIZE*4, false, true);

    const size_t ctx_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE*4 + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE*4, false);
    ggml_init_params ip = { ctx_size, NULL, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE*4, false);

    // Inputs live in their own context so they can be allocated on the CUDA buffer type,
    // exactly as a real model's activations would be.
    std::vector<ggml_tensor *> inputs;
    int expect_low_pad = 0;

    for (int il = 0; il < n_layer; ++il) {
        // shapes after the permute+cont in parakeet.cpp:165-167: [d_head, n_time, n_head]
        ggml_tensor * Q_u   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_head, n_time, n_head);
        ggml_tensor * K_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_head, n_time, n_head);
        ggml_tensor * V_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_head, n_time, n_head);
        // attn_scores before the expand-back pad: [window_size, n_time, n_head]
        ggml_tensor * scores = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, window_size, n_time, n_head);
        inputs.push_back(Q_u); inputs.push_back(K_cur);
        inputs.push_back(V_cur); inputs.push_back(scores);

        // parakeet.cpp:179-180 -- right pad the time dimension
        ggml_tensor * Q_u_padded = need_padding
            ? ggml_pad_ext(ctx, Q_u, 0, 0, 0, n_time_padded - n_time, 0, 0, 0, 0) : Q_u;

        // parakeet.cpp:184 -- pad front AND back: this is the low-side pad
        ggml_tensor * K_padded = ggml_pad_ext(ctx, K_cur, 0, 0, att_left, att_right, 0, 0, 0, 0);
        expect_low_pad++;
        // parakeet.cpp:185-186
        if (n_kv_dense > K_padded->ne[1]) {
            K_padded = ggml_pad_ext(ctx, K_padded, 0, 0, 0, n_kv_dense - K_padded->ne[1], 0, 0, 0, 0);
        }

        // parakeet.cpp:242 -- the same low-side pad for V
        ggml_tensor * V_padded = ggml_pad_ext(ctx, V_cur, 0, 0, att_left, att_right, 0, 0, 0, 0);
        expect_low_pad++;
        // parakeet.cpp:243-244
        if (n_kv_dense > V_padded->ne[1]) {
            V_padded = ggml_pad_ext(ctx, V_padded, 0, 0, 0, n_kv_dense - V_padded->ne[1], 0, 0, 0, 0);
        }

        // parakeet.cpp:228-232 -- expand probs back to n_kv_chunk width
        ggml_tensor * probs = need_padding
            ? ggml_pad_ext(ctx, scores, 0, 0, 0, n_time_padded - n_time, 0, 0, 0, 0) : scores;
        probs = ggml_reshape_4d(ctx, probs, window_size, chunk, n_group, n_head);
        probs = ggml_pad_ext(ctx, probs, 0, chunk, 0, 0, 0, 0, 0, 0);

        // Give every pad a CUDA-supported consumer so a CPU-assigned pad is a real boundary.
        ggml_build_forward_expand(gf, ggml_scale(ctx, Q_u_padded, 1.0f));
        ggml_build_forward_expand(gf, ggml_scale(ctx, K_padded,   1.0f));
        ggml_build_forward_expand(gf, ggml_scale(ctx, V_padded,   1.0f));
        ggml_build_forward_expand(gf, ggml_scale(ctx, probs,      1.0f));
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend_cuda);
    if (buf == NULL) {
        fprintf(stderr, "FATAL: could not allocate input tensors on CUDA0\n");
        return 2;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<float> ones(ggml_nelements(inputs[i]), 1.0f);
        ggml_backend_tensor_set(inputs[i], ones.data(), 0, ggml_nbytes(inputs[i]));
    }

    // The measurement is the SPLIT ASSIGNMENT, which ggml_backend_sched_reserve() performs in
    // full -- it is what decides whether an op falls back to the CPU. Executing the graph is
    // not needed to count that, and is deliberately opt-in: with the low pads declined, the
    // cross-backend copy path segfaults during compute in this tree (see the note below), and
    // a measurement that only survives in one of the two conditions is not a comparison.
    // Correctness of the kernel itself is established elementwise by test-backend-ops.
    if (!ggml_backend_sched_reserve(sched, gf)) {
        fprintf(stderr, "FATAL: sched reserve failed\n");
        return 2;
    }
    if (getenv("GGML_PAD_SCHED_COMPUTE") != NULL) {
        ggml_backend_sched_reset(sched);
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FATAL: graph compute failed\n");
            return 2;
        }
    }

    int n_nodes = ggml_graph_n_nodes(gf);
    int n_cpu = 0, n_pad = 0, n_pad_cpu = 0, n_low_pad = 0, n_low_pad_cpu = 0;
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, node);
        const bool on_cpu = (b == backend_cpu);
        if (on_cpu) n_cpu++;
        if (node->op == GGML_OP_PAD) {
            n_pad++;
            const bool low = node->op_params[0] || node->op_params[2] ||
                             node->op_params[4] || node->op_params[6];
            if (low) n_low_pad++;
            if (on_cpu) n_pad_cpu++;
            if (low && on_cpu) n_low_pad_cpu++;
        }
    }

    // A run that found no low-side pad at all would be measuring its own graph builder, not
    // the scheduler, so say so loudly rather than reporting a clean zero.
    if (n_low_pad != expect_low_pad) {
        fprintf(stderr, "CONTROL FAILED: built %d low-side pads but found %d in the graph\n",
                expect_low_pad, n_low_pad);
        return 3;
    }

    printf("n_layer=%d n_time=%d att_left=%d chunk=%d window_size=%d n_kv_dense=%d need_padding=%d\n",
            n_layer, n_time, att_left, chunk, window_size, n_kv_dense, (int) need_padding);
    printf("nodes=%d  cpu_assigned=%d  splits=%d\n",
            n_nodes, n_cpu, ggml_backend_sched_get_n_splits(sched));
    printf("pad_nodes=%d  pad_on_cpu=%d  low_side_pads=%d  low_side_pads_on_cpu=%d\n",
            n_pad, n_pad_cpu, n_low_pad, n_low_pad_cpu);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend_cuda);
    return 0;
}
