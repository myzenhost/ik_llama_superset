#pragma once

// Graph-level TurboQuant rotation for attention that runs on the CPU backend.
//
// A turbo K/V block is stored in a ROTATED basis (R = diag(S2) . H128 . diag(S1), orthogonal) and dequantises to that basis (ggml-turbo-quant.c).
// Attention in that basis is exact once  Q <- R Q  (K side)  and  out <- R^T out  (V side): (R q).(R k) == q.k and sum_i p_i (R v_i) == R (sum_i p_i v_i).
//
// WHO ROTATES. The CUDA, Vulkan, Metal and SYCL backends do it inside their own flash-attention wrapper, because their kernels only read the stored basis.
// The CPU backend has no such wrapper (its kernel is the generic one: turbo K through the f32 vec_dot of ggml-cpu.c, turbo V through to_float), so the graph
// carries GGML_OP_TURBO_WHT around the attention -- but ONLY when the cache lives in host memory, i.e. when the CPU backend runs the attention. A cache in a
// device buffer is attended by that device's wrapper, and rotating it here as well would rotate twice (silently wrong output, no error). One predicate,
// llama_turbo_graph_rotates(), is the single place this is decided; the Q side and the output side both go through it. The count per attention call is
// therefore: host cache -> 1 forward on Q (K turbo) + 1 inverse on the output (V turbo); device cache -> 0 in the graph, 1 + 1 inside the backend wrapper.
//
// Rotation groups are 128 wide (the turbo block), so the head size seen by the attention must be a multiple of 128; llama-context.cpp refuses the others.

#include "ggml.h"
#include "ggml-backend.h"

static inline bool llama_turbo_is_type(enum ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
}

// the allocated tensor behind a (possibly view) cache tensor
static inline const ggml_tensor * llama_turbo_root(const ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

// true when `cache` is a turbo tensor whose data lives in host memory, i.e. the CPU backend will run the attention that reads it
static inline bool llama_turbo_graph_rotates(const ggml_tensor * cache) {
    const ggml_tensor * r = llama_turbo_root(cache);
    return r && llama_turbo_is_type(r->type) && r->buffer && ggml_backend_buffer_is_host(r->buffer);
}

// Q for the attention, rotated into the K cache's basis when the graph has to do it. q: [head, n_tokens, n_head, ...].
static inline ggml_tensor * llama_turbo_rotate_q(ggml_context * ctx, ggml_tensor * q, const ggml_tensor * k_cache) {
    if (!llama_turbo_graph_rotates(k_cache)) {
        return q;
    }
    GGML_ASSERT(q->ne[0] % 128 == 0 && "turbo rotation groups are 128 wide");
    if (q->type != GGML_TYPE_F32) {
        q = ggml_cast(ctx, q, GGML_TYPE_F32);
    }
    if (!ggml_is_contiguous(q)) {
        q = ggml_cont(ctx, q);
    }
    return ggml_turbo_wht(ctx, q, nullptr, 0, 128);   // 0 = forward
}

// attention output, rotated back out of the V cache's basis when the graph has to do it. cur: [head_v, n_head, n_tokens, ...], F32.
static inline ggml_tensor * llama_turbo_unrotate_out(ggml_context * ctx, ggml_tensor * cur, const ggml_tensor * v_cache) {
    if (!llama_turbo_graph_rotates(v_cache)) {
        return cur;
    }
    GGML_ASSERT(cur->ne[0] % 128 == 0 && "turbo rotation groups are 128 wide");
    if (!ggml_is_contiguous(cur)) {
        cur = ggml_cont(ctx, cur);
    }
    return ggml_turbo_wht(ctx, cur, nullptr, 1, 128);   // 1 = inverse
}
