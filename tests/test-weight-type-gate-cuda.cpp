// Per-type, per-device table of what the weight-type placement gate asks.
//
// src/llama-load-tensors.cpp create_tensors_helper::buft_runs_weight decides, for each quantized weight, whether the
// device behind a buffer type can run the op that consumes it (GET_ROWS for token_embd, MUL_MAT_ID for 3-D expert
// stacks, MUL_MAT otherwise); "no" keeps the weight in the CPU buffer. That helper is a private member of an internal
// loader class and needs a loader, so it cannot be called from a test: this file MIRRORS its probe (same tensors, same
// zero-size WEIGHTS-usage buffer binding, same shapes except ne[0] is fixed to a multiple of every block size) over the
// public ggml backend API, and prints the answer for every device and type. Keep it in step with buft_runs_weight.
//
// Output: "TYPE OP supported|unsupported" per device. Exit 0 unless a probe crashes; 77 = SKIPPED (no non-CPU device).
// Row S69-ENGREV-UPSTREAM-MERGE-TENSOR-INFOS-TABLE-DECIDE-WITH-LOADER-2026-11-04-01. Builds CPU-only.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

enum probe_op { OP_GET_ROWS, OP_MUL_MAT_ID, OP_MUL_MAT, OP_COUNT };
static const char * const op_name[OP_COUNT] = { "GET_ROWS", "MUL_MAT_ID", "MUL_MAT" };

// Mirrors buft_runs_weight for ONE consumer op. 1 = supported, 0 = unsupported, -1 = probe could not be built.
static int probe(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft, ggml_type type, probe_op op) {
    ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead() * 16, /*.mem_buffer =*/ NULL, /*.no_alloc =*/ true };
    ggml_context * tctx = ggml_init(ip);
    if (!tctx) {
        return -1;
    }
    const int64_t ne0 = 4096;   // multiple of every block size (max 256)
    const int64_t ne1 = 64;
    const int64_t ne2 = op == OP_MUL_MAT_ID ? 4 : 1;
    const char * name = op == OP_GET_ROWS ? "token_embd.weight" : op == OP_MUL_MAT_ID ? "blk.0.ffn_up_exps.weight" : "blk.0.attn_q.weight";
    ggml_tensor * w = ggml_new_tensor_3d(tctx, type, ne0, ne1, ne2);
    ggml_set_name(w, name);
    ggml_backend_buffer_t dummy = ggml_backend_buft_alloc_buffer(buft, 0);
    if (dummy) {
        ggml_backend_buffer_set_usage(dummy, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        w->buffer = dummy;
        w->data   = (void *)(uintptr_t) ggml_backend_buffer_get_alignment(dummy);
    }
    bool ok;
    if (op == OP_GET_ROWS) {
        ggml_tensor * ids = ggml_new_tensor_1d(tctx, GGML_TYPE_I32, 1);
        ok = ggml_backend_dev_supports_op(dev, ggml_get_rows(tctx, w, ids));
    } else if (op == OP_MUL_MAT_ID) {
        ggml_tensor * b   = ggml_new_tensor_3d(tctx, GGML_TYPE_F32, ne0, 1, 1);
        ggml_tensor * ids = ggml_new_tensor_2d(tctx, GGML_TYPE_I32, 1, 1);
        ok = ggml_backend_dev_supports_op(dev, ggml_mul_mat_id(tctx, w, b, ids));
    } else {
        ggml_tensor * b = ggml_new_tensor_4d(tctx, GGML_TYPE_F32, ne0, 512, ne2, 1);
        ok = ggml_backend_dev_supports_op(dev, ggml_mul_mat(tctx, w, b));
    }
    w->buffer = nullptr;
    if (dummy) {
        ggml_backend_buffer_free(dummy);
    }
    ggml_free(tctx);
    return ok ? 1 : 0;
}

int main() {
    ggml_backend_load_all();

    const size_t n_dev = ggml_backend_dev_count();
    int n_non_cpu = 0;
    for (size_t i = 0; i < n_dev; ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            ++n_non_cpu;
        }
    }
    printf("devices: %zu total, %d non-CPU\n", n_dev, n_non_cpu);
    if (n_non_cpu == 0) {
        printf("SKIP: no non-CPU device\n");
        return 77;
    }

    int n_fail_quant = 0;       // quantized (type, op) pairs a non-CPU device refuses: these weights would stay on CPU
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        printf("== device %zu: %s (%s) type=%d buft=%s\n", i, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                (int) ggml_backend_dev_type(dev), buft ? ggml_backend_buft_name(buft) : "(null)");
        if (!buft) {
            continue;
        }
        int n_unsup_dev = 0;
        for (int t = 0; t < (int) GGML_TYPE_COUNT; ++t) {
            const ggml_type type = (ggml_type) t;
            const ggml_type_traits * tr = ggml_get_type_traits(type);
            if (!tr || !tr->type_name || tr->blck_size <= 0 || (4096 % tr->blck_size) != 0) {
                continue;
            }
            const bool quant = ggml_is_quantized(type);
            const bool floaty = type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
            if (!quant && !floaty) {
                continue;       // integer/aux types are never weights
            }
            for (int o = 0; o < OP_COUNT; ++o) {
                const int r = probe(dev, buft, type, (probe_op) o);
                printf("%-10s %-10s %s%s\n", tr->type_name, op_name[o],
                        r < 0 ? "probe-error" : r ? "supported" : "unsupported", quant ? "" : "  (float, not gated)");
                if (r == 0 && quant && !is_cpu) {
                    ++n_unsup_dev;
                }
            }
        }
        if (!is_cpu) {
            printf("-- device %s: %d quantized (type,op) pairs unsupported (the gate keeps such weights on CPU)\n",
                    ggml_backend_dev_name(dev), n_unsup_dev);
            n_fail_quant += n_unsup_dev;
        }
    }
    printf("TOTAL quantized (type,op) pairs unsupported on non-CPU devices: %d\n", n_fail_quant);
    return 0;
}
