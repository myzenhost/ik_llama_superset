#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dequantize two HF integer per-row checkpoint formats for the HF->GGUF converter.

Both produce float32 and let llama-quantize re-quantize afterwards (lossy fallback route, like conversion/bnb4.py).
Nothing here is copied: the formats were read from the sources named below and the maths re-derived.

1. quant_method "gemma"  (transformers GemmaQuantizationConfig / GemmaQuantizer, Apache-2.0)
   SOURCE-READ transformers @ 95bafdcccfffe17ec383999fc39273867b11385e, no code copied:
     src/transformers/integrations/gemma_quant.py:39-50   _unpack_int4: low nibble = first value, stored 0..15, value = n - 8
     src/transformers/integrations/gemma_quant.py:53-64   _unpack_int2: lane k = bits [2k+1:2k], first value in the LOW lane, value = n - 2
     src/transformers/integrations/gemma_quant.py:80-95   QuantizedLinear storage: int2 uint8 [out, ceil(in/4)], int4 uint8
                                                          [out, ceil(in/2)], int8 int8 [out, in]; weight_scale float32 [out, 1]
     src/transformers/integrations/gemma_quant.py:97-108  W = unpack(weight) * weight_scale   (per OUTPUT row)
     src/transformers/integrations/gemma_quant.py:122-179 QuantizedEmbedding: embedding_quantized [N, packed] +
                                                          embedding_scale [N, nblocks]; block = dim // nblocks;
                                                          row = unpack * repeat_interleave(scale, block)
     src/transformers/utils/quantization_config.py:2041-2072  num_bits (default 4), quantize_embeddings, module_quant_configs
                                                          (regex via re.search -> {"num_bits": 2|4|8}), modules_to_not_convert
     src/transformers/quantizers/quantizer_gemma.py:58-63   k_cache_scale / v_cache_scale are in the file but used by nothing
   Dropped with a warning (no ggml home): input_activation_scale / output_activation_scale (SRQ activation rounding),
   k_cache_scale / v_cache_scale.
   Limitation: a packed int4/int2 row carries no true width, so the unpacked width is packed*(2|4); a checkpoint whose
   in_features is odd (int4) / not a multiple of 4 (int2) holds padding that this reader cannot see for LINEAR weights.
   For embeddings a width that is not a multiple of the scale block count is rejected.

2. bitsandbytes LLM.int8 (quantization_config.load_in_8bit)
   SOURCE-READ bitsandbytes 0.45.5 (MIT) and transformers @ 95bafdc (Apache-2.0), no code copied:
     bitsandbytes/functional.py:2738-2749    int8_vectorwise_dequant: A * stats.view(-1, 1) * 7.874015718698502e-3  (= 1/127)
     bitsandbytes/nn/modules.py:912-934      Linear8bitLt._save_to_state_dict: <p>.weight int8, <p>.SCB float32 [out],
                                             <p>.weight_format uint8 scalar 0 ("row", the only format written now)
     bitsandbytes/nn/modules.py:644-657      weight_format other than "row" = legacy tile layout (needs get_tile_inds)
     transformers/integrations/bitsandbytes.py:257-267   same dequant as above
   Outlier columns (llm_int8_threshold) are a RUN-TIME split of the same int8 matrix, so full dequant is the whole weight.
"""
from __future__ import annotations

import logging
import re

import torch

logger = logging.getLogger("hf-to-gguf")

BNB_INT8_RECIPROCAL = 7.874015718698502e-3  # bitsandbytes' own constant (1/127 as float32-rounded double)

# tensors that exist only for run-time activation / kv-cache quantization
GEMMA_DROP_SUFFIXES = (".input_activation_scale", ".output_activation_scale", ".k_cache_scale", ".v_cache_scale")


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [.., n] -> int8 [.., 2n]; low nibble first, value = nibble - 8. Python-int shifts keep lazy/meta tensors working."""
    packed = packed.to(torch.uint8)
    low = (packed & 0x0F).to(torch.int8) - 8
    high = (packed >> 4).to(torch.int8) - 8
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def unpack_int2(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [.., n] -> int8 [.., 4n]; lane 0 (bits 1:0) first, value = lane - 2."""
    packed = packed.to(torch.uint8)
    lanes = [((packed >> s) & 0x03).to(torch.int8) - 2 for s in (0, 2, 4, 6)]
    return torch.stack(lanes, dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4)


def gemma_num_bits(name: str, weight: torch.Tensor, quant_config: dict) -> int:
    """Bit width of one tensor: int8 storage is 8; packed uint8 is 2 or 4 from module_quant_configs (re.search), else num_bits."""
    if weight.dtype == torch.int8:
        return 8
    bits = quant_config.get("num_bits", 4)
    # module_quant_configs is ordered; the first matching pattern wins (transformers gemma_quant.py:221-227)
    for pattern, override in (quant_config.get("module_quant_configs") or {}).items():
        if re.search(pattern, name):
            bits = (override or {}).get("num_bits", bits)
            break
    if bits not in (2, 4):
        raise NotImplementedError(f"gemma quant: tensor {name!r} is uint8-packed but resolves to num_bits={bits}; only 2 or 4 pack into uint8")
    return bits


def _unpack(weight: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return weight
    return unpack_int4(weight) if bits == 4 else unpack_int2(weight)


def dequant_gemma_linear(weight: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """W[o, :] = unpack(weight[o, :]) * weight_scale[o]  -> float32."""
    scale = scale.float()
    if scale.ndim == 1:
        scale = scale.reshape(-1, 1)
    return _unpack(weight, bits).float() * scale


def dequant_gemma_embedding(quantized: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """row = unpack(q) * repeat_interleave(embedding_scale, dim // nblocks)  -> float32 (embed_scale is applied by the model, not here)."""
    rows = _unpack(quantized, bits).float()
    nblocks = scale.shape[-1]
    dim = rows.shape[-1]
    if dim % nblocks != 0:
        raise NotImplementedError(
            f"gemma quant: embedding width {dim} is not a multiple of the {nblocks} scale blocks (padded int4/int2 packing?)")
    return rows * scale.float().repeat_interleave(dim // nblocks, dim=-1)


def dequant_bnb_int8(weight: torch.Tensor, scb: torch.Tensor) -> torch.Tensor:
    """LLM.int8 row-wise: W[o, :] = int8[o, :] * SCB[o] / 127  -> float32."""
    return weight.float() * scb.float().reshape(-1, 1) * BNB_INT8_RECIPROCAL
