#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

//
// llama_control_vector
//
// MOVED HERE 2026-09-09, slice 26, from src/llama-context.h:420. Placed beside llama_adapter_cvec
// deliberately, so that the fact THIS FORK CARRIES TWO CONTROL-VECTOR TYPES is visible in one
// screen instead of split across headers.
//
// The difference that matters is which one has a PRODUCER. Nothing in this fork ever constructs a
// llama_adapter_cvec: measured, its 12 occurrences in src/ are its own definition, its own
// implementation, its ptr alias, two pointer-typed fields in llama-graph.h, and comments -- zero
// constructions. This type is the one that gets filled, by llama_control_vector_init and
// llama_control_vector_apply (src/llama.cpp:9922, :9978), which is also what
// llama_context::set_adapter_cvec calls since slice 23.
//
// It moved here so src/llama-graph.h can name it in llm_graph_params. llama-context.h, where it
// used to live, already includes llama-graph.h, so llama-graph.h could not include that header back
// without a cycle; llama-adapter.h is already included by llama-graph.h at line 6 and supplies
// everything this struct needs (<vector>, and ggml-cpp.h for ggml_add / ggml_free /
// ggml_backend_buffer_free).
struct llama_control_vector {
    std::vector<struct ggml_tensor *> tensors; // per layer
    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    struct ggml_tensor * tensor_for(int il) const {
        if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
            return nullptr;
        }
        return tensors[il];
    }

    struct ggml_tensor * apply_to(struct ggml_context * ctx, struct ggml_tensor * cur, int  il) const {
        ggml_tensor * layer_dir = tensor_for(il);
        if (layer_dir != nullptr) {
            // ik a4ca26a5d (#2660), ported here because this fork's control vector lives in
            // llama-adapter.h, not llama-context.h: with split-mode graph and a bf16 reduce type
            // `cur` can arrive non-F32 while the direction tensor is F32.
            if (layer_dir->type != cur->type && layer_dir->type == GGML_TYPE_F32) {
                cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
            }
            cur = ggml_add(ctx, cur, layer_dir);
        }
        return cur;
    }

    ~llama_control_vector() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
    }
};

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // DoRA runtime kind (row S69-ADAPTERS-DORA-RUNTIME-KIND-2026-10-26-01): d - 1 per OUTPUT row of the base weight, F32 [n_out]
    // (or [n_out, n_expert] for stacked experts), with d = m / ||W + s B A||_row computed ONCE at adapter load from the
    // dequantised base (s = get_scale(alpha, 1)). nullptr = plain LoRA. The graph adds, per adapter, PEFT's additive form
    // (d - 1) * res + d * s B(A x) = (res + s B(A x)) * (d - 1) + s B(A x), before the caller's bias (peft 0.17.1
    // tuners/lora/dora.py:87-99; variants.py:110-130 passes the running result as base_result).
    ggml_tensor * dora_dm1 = nullptr;

    // Lily runtime kind (row S69-ADAPTERS-W2-LILY-PER-SEQUENCE-ROUTER-2026-10-26-01): router R, ggml [r, num_B]. nullptr = not Lily.
    // For a Lily weight a = A [n_in, r], b = the num_B up-projections [r, n_out, num_B] with PEFT's `scaling` folded in by the
    // converter, and the delta of token t is  sum_e c[e, t] * B_e (A x_t),  c = mean over the tokens of t's sequence in the
    // ubatch of softmax(R A x). PEFT averages over EVERY token of its forward batch instead (peft a7e8ea4a1227
    // tuners/lily/layer.py:150-179), so it equals this only at batch 1 -- the per-sequence mean is the point of the kind.
    ggml_tensor * lily_router = nullptr;

    // Embedding layout (token_embd): a = [r, n_vocab] rows looked up by token id, b = [r, n_embd]; the delta of token t is
    // s * B (A[:, t]) added to the embedding row (peft lora/layer.py Embedding.forward). Never a linear LoRA: a tied lm_head
    // shares the tensor NAME but must not use this pair (row S69-ENGREV-PEFT-EMBED-LORA-NOT-APPLIED-LEGACY-2026-10-27-01).
    bool is_embd = false;

    // IA3 runtime kind (row S69-ADAPTERS-LINEAR-WRAPPER-IA3-RED-VEC-SCALE-2026-10-26-01): learned elementwise activation scales, F32.
    // ia3_in [n_in]  multiplies the INPUT of the matmul (PEFT IA3 "feedforward" modules: result = base(x * l), peft a7e8ea4a1227
    // tuners/ia3/layer.py:172-201); ia3_out [n_out] multiplies the OUTPUT (result = base(x) * l). A weight is either LoRA-shaped
    // (a, b set) or IA3-shaped (a == b == nullptr and exactly one of ia3_in / ia3_out set); never both.
    ggml_tensor * ia3_in  = nullptr;
    ggml_tensor * ia3_out = nullptr;
    bool is_ia3() const { return ia3_in != nullptr || ia3_out != nullptr; }

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b, ggml_tensor * dora_dm1 = nullptr, ggml_tensor * lily_router = nullptr)
        : a(a), b(b), dora_dm1(dora_dm1), lily_router(lily_router) {}
};

// Lily delta of ONE adapter on one 2-D linear, before any adapter scale / per-sequence mask (row
// S69-ADAPTERS-W2-LILY-PER-SEQUENCE-ROUTER-2026-10-26-01). cur [n_in, T]; seq_mean [T, T] F32 with seq_mean[k, j] = 1/n(seq j)
// when token k is in the same sequence as token j, else 0. Returns [n_out, T].
//   h = A x                      [r, T]
//   p = softmax(R h)             [num_B, T]
//   c = p @ seq_mean             [num_B, T]   (per-sequence mean of p, broadcast back to every token of the sequence)
//   y = B_all h                  [n_out, num_B, T]
//   delta[:, t] = y[:, :, t] c[:, t]
inline ggml_tensor * llama_lily_delta(ggml_context * ctx, const llama_adapter_lora_weight & lw, ggml_tensor * cur, ggml_tensor * seq_mean) {
    const int64_t r     = lw.a->ne[1];
    const int64_t n_out = lw.b->ne[1];
    const int64_t n_b   = lw.b->ne[2];
    const int64_t T     = cur->ne[1];
    ggml_tensor * h = ggml_mul_mat(ctx, lw.a, cur);                                       // [r, T]
    ggml_tensor * p = ggml_soft_max(ctx, ggml_mul_mat(ctx, lw.lily_router, h));         // [num_B, T]
    ggml_tensor * c = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, p)), seq_mean); // [num_B, T]
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, lw.b, r, n_out * n_b), h);   // [n_out * num_B, T]
    y = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, y, n_out, n_b, T), 1, 0, 2, 3)); // [num_B, n_out, T]
    ggml_tensor * d = ggml_mul_mat(ctx, y, ggml_reshape_3d(ctx, c, n_b, 1, T));          // [n_out, 1, T]
    return ggml_reshape_2d(ctx, d, n_out, T);
}

// DoRA delta of ONE adapter on one linear (row S69-ADAPTERS-DORA-RUNTIME-KIND-2026-10-26-01), PEFT additive form:
//   delta = (d - 1) * res + d * lora = (res + lora) * (d - 1) + lora
// res  = the linear's output so far, BEFORE the caller's bias (base product + earlier adapters' deltas: PEFT passes its running
//        `result` as base_result, peft 0.17.1 tuners/lora/variants.py:110-130, and subtracts the bias, dora.py:91-95)
// lora = s * B(A x), already scaled by s = get_scale(alpha, 1)
// ids  = nullptr for a 2-D linear (dm1 [n_out] broadcasts over tokens); for stacked experts (res [n_out, n_used, n_tokens]) the
//        expert ids of ggml_mul_mat_id, used to gather dm1 [n_out, n_expert] per (slot, token).
// The caller applies the adapter scale c (c != 1) or the per-sequence mask to the returned delta, never inside it.
inline ggml_tensor * llama_dora_delta(ggml_context * ctx, const llama_adapter_lora_weight & lw, ggml_tensor * res, ggml_tensor * lora,
                                      ggml_tensor * ids = nullptr) {
    ggml_tensor * dm1 = lw.dora_dm1;
    if (ids != nullptr) {
        ggml_tensor * idf = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx, ids);
        idf = ggml_reshape_1d(ctx, idf, ggml_nelements(ids));
        dm1 = ggml_get_rows(ctx, dm1, idf);                                  // [n_out, n_used * n_tokens]
        dm1 = ggml_reshape_3d(ctx, dm1, res->ne[0], res->ne[1], res->ne[2]); // [n_out, n_used, n_tokens]
    }
    return ggml_add(ctx, ggml_mul(ctx, ggml_add(ctx, res, lora), dm1), lora);
}

// IA3 scale vector at adapter scale c: c = 1 is PEFT-exact (l), c = 0 is off (1), between is the linear blend 1 + c (l - 1)
// (the same "mult" reading the DoRA kind uses). Returns a [n] tensor that broadcasts against an [n, T] activation.
inline ggml_tensor * llama_ia3_vec(ggml_context * ctx, ggml_tensor * l, float c) {
    return c == 1.0f ? l : ggml_scale_bias(ctx, l, c, 1.0f - c);
}

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // adapter.kinds lists "lily": every weight carries a router (llama_adapter_lora_weight::lily_router)
    bool kind_lily = false;

    // adapter.lora.gate: which token rows receive the delta (row S69-ADAPTERS-POSITION-GATED-LORA-UNO-SPEC-2026-10-26-01).
    // draft_rows = tokens at/after the sequence's llama_set_draft_start position; mask_token = tokens equal to gate_token.
    enum llama_lora_gate { LLAMA_LORA_GATE_NONE = 0, LLAMA_LORA_GATE_DRAFT_ROWS, LLAMA_LORA_GATE_MASK_TOKEN };
    llama_lora_gate gate = LLAMA_LORA_GATE_NONE;
    llama_token     gate_token = -1;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        uint32_t n = 0;
        for (const auto & it : ab_map) {
            if (it.second.is_ia3()) {
                n += 4u; // scale vector (scale_bias), mul (+ spare)
                continue;
            }
            n += 6u; // a, b, scale, add, 2 x mul_mat
            if (gate != LLAMA_LORA_GATE_NONE) {
                n += 2u; // gate mask (+ get_rows after output pruning), mul
            }
            if (it.second.dora_dm1) {
                n += 8u; // dm1, add, mul, add (+ scale/mask, expert gather: cont, reshape, get_rows, reshape)
            }
            if (it.second.lily_router) {
                n += 16u; // router, mul_mat, soft_max, transpose, cont, mul_mat, reshape, mul_mat, reshape, permute, cont, reshape, mul_mat, reshape (+ scale/mask)
            }
        }
        return n;
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;

// Per-SEQUENCE adapters (heterogeneous batch): sequence -> list of (adapter, scale). Applied ADDITIVELY to the context-wide
// llama_adapter_loras. `adapters` is the union over all sequences, in first-seen order; it alone decides the graph topology
// (one masked low-rank branch per adapter), `epoch` is bumped when that union changes so a cached graph is rebuilt. The
// per-sequence scales are read at set_input() time, so changing only the scales (or which sequence uses which adapter)
// reuses the graph.
struct llama_seq_lora_state {
    std::vector<llama_adapter_lora *> adapters;
    std::map<llama_seq_id, std::vector<std::pair<llama_adapter_lora *, float>>> by_seq;
    uint64_t epoch = 0;
};
