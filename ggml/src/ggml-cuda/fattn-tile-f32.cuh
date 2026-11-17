#pragma once

#include "common.cuh"

void ggml_cuda_flash_attn_ext_tile_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_fattn_tile_f32_is_supported(int device, const ggml_tensor * dst);

// ik: Pascal head-size-512 tile kernel (fattn-tile-new.cu)
void ggml_cuda_flash_attn_ext_tile_new(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_fattn_tile_new_is_supported(int device, const ggml_tensor * dst);
