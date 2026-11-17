// offload_attention_bridge.cpp — reach the #220 offload-attention path from a live llama_context.
//
// offload_attention.cpp implements the math against a plain OffloadKVStore (raw float K/V and a
// token count). This file is the adapter that builds such a store from the running KV cache, so
// the capability is reachable from serving code instead of only from a unit test.
//
// It follows triattention-bridge.cpp exactly: same get_kv() shape, same extern "C" surface, same
// use of llama_kv_cache_fork::get_k_raw(layer). That bridge is the fork's established way to read KV
// from a context, and reusing it means this adds no new access pattern to review.
//
// WHAT IT REFUSES, AND WHY THAT IS THE POINT.
// The KV cache is not guaranteed to be host-resident F32. It is commonly F16 or a quantised type,
// and with -ngl it lives on the device. The offload-attention math needs contiguous float keys and
// values. Rather than cast a quantised or device pointer and produce confident garbage, every
// unsupported case returns a DISTINCT negative code and touches nothing:
//   -1 bad argument            -4 V tensor unavailable
//   -2 K tensor unavailable    -5 K/V type is not F32 (F16 or quantised -- needs a dequant path)
//   -3 empty KV                -6 tensor data is not host-accessible (device-resident KV)
// A caller can tell "not applicable here" from "it ran", which a single 0/-1 could not. This is
// the same failure the port itself was fixing one level up: a thing that looks wired and silently
// does nothing.
//
// STATED LIMIT: the F32-host case is the one this serves today. F16/quantised and device-resident
// KV need a dequant-and-stage step, which belongs with the CUDA gather kernel (#220 kernel
// compose) rather than here -- doing it in the bridge would mean a full KV copy per step, which is
// the cost the feature exists to avoid.

#include "llama.h"
#include "ggml-backend.h"
#include "llama-context.h"
#include "offload_attention.h"

#include <vector>

namespace {

llama_kv_cache_fork * get_kv(void * ctx_void) {
    if (!ctx_void) return nullptr;
    auto * ctx = static_cast<llama_context *>(ctx_void);
    return &ctx->kv_self;
}


// Is this tensor's memory readable by host code?
//
// MEASURED 2026-08-19, and it invalidated the previous check. The original test was
// `t->data == nullptr`. A CUDA-resident tensor has a NON-NULL data pointer -- it is a DEVICE
// address -- so that test passed, the probe returned OK, and the bridge dereferenced device
// memory from host code. Result: 0xC0000005 access violation with n_gpu_layers=99, on a context
// that reported "llama_kv_cache_init: CUDA0 KV buffer size = 22.50 MiB". The NOT_HOST refusal I
// had reasoned about could never have fired.
//
// ggml_backend_buffer_is_host() asks the buffer what it is, instead of inferring residency from
// a pointer value. A tensor with no buffer at all is treated as not-host: unknown provenance is
// not a licence to dereference.
static bool tensor_is_host(const struct ggml_tensor * t) {
    if (!t || !t->data) return false;
    if (!t->buffer)     return false;
    return ggml_backend_buffer_is_host(t->buffer);
}

}  // namespace

extern "C" {

// Status codes, exposed so a caller can branch on the reason rather than on "nonzero".
#define OFFLOAD_ATTN_OK              0
#define OFFLOAD_ATTN_E_ARG          -1
#define OFFLOAD_ATTN_E_NO_K         -2
#define OFFLOAD_ATTN_E_EMPTY_KV     -3
#define OFFLOAD_ATTN_E_NO_V         -4
#define OFFLOAD_ATTN_E_TYPE         -5
#define OFFLOAD_ATTN_E_NOT_HOST     -6

// Report whether offload-attention can run against this context/layer as it currently stands.
// Cheap: inspects tensor metadata only, allocates nothing, and is safe to call per request.
int offload_attn_probe(void * ctx_void, int layer_idx) {
    llama_kv_cache_fork * kv = get_kv(ctx_void);
    if (!kv || layer_idx < 0) return OFFLOAD_ATTN_E_ARG;

    struct ggml_tensor * k = kv->get_k_raw(layer_idx);
    if (!k) return OFFLOAD_ATTN_E_NO_K;
    if (kv->get_used_n_kv() == 0) return OFFLOAD_ATTN_E_EMPTY_KV;
    if (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16) return OFFLOAD_ATTN_E_TYPE;
    if (!tensor_is_host(k)) return OFFLOAD_ATTN_E_NOT_HOST;
    return OFFLOAD_ATTN_OK;
}

// One offload-attention decode step over layer `layer_idx` of the live KV cache.
//   q        [head_dim] query for a single head
//   out      [head_dim] receives the attention output
//   page_size / page_budget / selector map onto ik_offload::OffloadConfig
// Returns OFFLOAD_ATTN_OK, or one of the negative codes above with `out` left untouched.
int offload_attn_step_from_ctx(void * ctx_void, int layer_idx,
                               const float * q, float * out,
                               int page_size, float page_budget, int selector) {
    if (!q || !out) return OFFLOAD_ATTN_E_ARG;

    const int probe = offload_attn_probe(ctx_void, layer_idx);
    if (probe != OFFLOAD_ATTN_OK) return probe;

    llama_kv_cache_fork * kv = get_kv(ctx_void);
    struct ggml_tensor * k = kv->get_k_raw(layer_idx);

    // V is read through the same cache; if the fork ever splits its accessor this is the one
    // place to update. Absent V is reported distinctly rather than treated as "no KV".
    struct ggml_tensor * v = (layer_idx < (int) kv->v_l.size()) ? kv->v_l[layer_idx] : nullptr;
    if (!v) return OFFLOAD_ATTN_E_NO_V;
    if (v->type != GGML_TYPE_F32 && v->type != GGML_TYPE_F16) return OFFLOAD_ATTN_E_TYPE;
    if (!tensor_is_host(v)) return OFFLOAD_ATTN_E_NOT_HOST;

    // Row layout matches what triattention-runtime.c asserts: ne[0] == n_kv_head * head_dim per
    // cell, one row per cell. head_dim is therefore ne[0] for a single-head view.
    ik_offload::OffloadKVStore store;
    store.n_tokens  = (int64_t) kv->get_used_n_kv();
    store.head_dim  = k->ne[0];
    store.n_kv_head = 1;
    store.k_backing = k->data;
    store.v_backing = v->data;
    // F16 is the fork default (src/llama.cpp:4906); F32 KV is not a reachable config here,
    // MEASURED 2026-08-19 -- it is refused without flash_attn and aborts the FA kernel with it.
    store.k_type = (k->type == GGML_TYPE_F16) ? ik_offload::KVType::F16 : ik_offload::KVType::F32;
    store.v_type = (v->type == GGML_TYPE_F16) ? ik_offload::KVType::F16 : ik_offload::KVType::F32;

    ik_offload::OffloadConfig cfg;
    cfg.page_size   = page_size  > 0   ? page_size   : 16;
    cfg.page_budget = page_budget > 0.0f ? page_budget : 0.02f;
    switch (selector) {
        case 1:  cfg.selector = ik_offload::Selector::SHADOWKV; break;
        case 2:  cfg.selector = ik_offload::Selector::RETATTN;  break;
        default: cfg.selector = ik_offload::Selector::QUEST;    break;
    }

    return ik_offload::offload_attend_step(q, store, cfg, out) == 0
               ? OFFLOAD_ATTN_OK
               : OFFLOAD_ATTN_E_ARG;
}

}  // extern "C"
