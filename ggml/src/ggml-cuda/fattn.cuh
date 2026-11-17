#pragma once

#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Takes a DEVICE INDEX, not a backend context. Its only caller is supports_op, which is a member
// of ggml_backend_device_i now and so has no backend context to give. Nothing was lost: the
// context was forwarded down the chain and read by nobody (both leaves marked it
// [[maybe_unused]]), while the only device-derived value, cc, came from ggml_cuda_get_device().
bool ggml_cuda_fattn_is_supported(int device, const ggml_tensor * dst);
