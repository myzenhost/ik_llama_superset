// Compatibility shim: force-included (/FI) ahead of every upstream TU in the shim probe.
//
// PURPOSE. The symbol-gap probe found only 17 distinct identifiers that upstream's
// KV-cache/graph layer needs and our tree does not provide. Seventeen is small enough
// that "the graft is blocked" would be a lazy conclusion without testing the obvious
// fix. This header supplies all 17 -- as enum constants, function declarations, and
// typedefs -- so the compile probe can answer a falsifiable question:
//
//     does providing the missing symbols let the layer compile, or does a second,
//     deeper wave of errors appear behind them?
//
// If it compiles, the graft is viable through a shim and the scope verdict flips.
// If a deeper wave appears, that is an EVIDENCED negative rather than an impression.
//
// NOTHING HERE IS A REAL IMPLEMENTATION. These are declarations to get past the
// front end. No shim object is linked; this probe is compile-only (cl /c).

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

// ---- ggml op / hint extensions our ggml generation lacks --------------------
#ifndef GGML_HINT_SRC0_IS_HADAMARD
#define GGML_HINT_SRC0_IS_HADAMARD 1
#endif
#ifndef GGML_TYPE_NVFP4
#define GGML_TYPE_NVFP4 ((enum ggml_type)(GGML_TYPE_COUNT + 1))
#endif

extern "C" {
void          ggml_mul_mat_set_hint(struct ggml_tensor * t, int hint);
struct ggml_tensor * ggml_argsort_top_k(struct ggml_context * ctx,
                                        struct ggml_tensor * a, int k);
struct ggml_tensor * ggml_swiglu_clamp(struct ggml_context * ctx,
                                       struct ggml_tensor * a,
                                       struct ggml_tensor * b, float limit);

// ---- ggml-backend DEVICE API (the generation gap) --------------------------
typedef ggml_backend_buffer_type_t (*ggml_backend_dev_get_extra_bufts_t)(
        ggml_backend_dev_t device);
ggml_backend_dev_t         ggml_backend_dev_by_type(enum ggml_backend_dev_type type);
ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device);
}

// ---- llama public API values our header lacks ------------------------------
#ifndef LLAMA_POOLING_TYPE_RANK
#define LLAMA_POOLING_TYPE_RANK ((enum llama_pooling_type)4)
#endif
#ifndef LLAMA_STATE_SEQ_FLAGS_ON_DEVICE
#define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE 2
#endif
