// Minimal ggml stub for the standalone KVFlash pager selftest (S69 #66).
// Provides ONLY what kvflash_pager.h touches: ggml_tensor (ne/nb/data/type).
// The real build uses the genuine ggml.h; this lets the pager LOGIC be proven
// host-side (gate C/E) without the full ik_llama build, exactly as the SnapKV
// core was standalone-compiled before wiring.
#pragma once
#include <cstdint>
#include <cstddef>

typedef enum { GGML_TYPE_F16 = 1, GGML_TYPE_Q8_0 = 8 } ggml_type;

struct ggml_tensor {
    ggml_type type;
    int64_t   ne[4];   // dims: [head_dim, pool_tokens, n_head_kv, 1]
    size_t    nb[4];   // strides in bytes
    void *    data;    // backing bytes (the stub treats this as "device" memory)
};
