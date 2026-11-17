#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dequantize HuggingFace bitsandbytes 4-bit (NF4 / FP4) checkpoints for the HF->GGUF converter.

FORMAT (written from the format, nothing copied; sources read in full 2026-10-05:
bitsandbytes/functional.py QuantState.as_dict/from_dict + dequantize_4bit + dequantize_blockwise,
bitsandbytes/utils.py pack_dict_to_tensor/unpack_tensor_to_dict (MIT), and the same layout in
onnxruntime contrib_ops/cpu/quantization/blockwise_quant_block_bnb4.h (MIT)):

  <prefix>.weight                                   uint8, two 4-bit codes per byte, FIRST element in the HIGH nibble
  <prefix>.weight.absmax                            one scale per `blocksize` elements of the FLATTENED weight
                                                    (float32; or uint8 codes when double-quantised)
  <prefix>.weight.quant_map                         the 16-entry codebook (NF4 or FP4) -- stored IN the checkpoint
  <prefix>.weight.nested_absmax / nested_quant_map  only with double quantisation: second-level scale and 256-entry
                                                    codebook used to dequantise `absmax`
  <prefix>.weight.quant_state.bitsandbytes__{nf4|fp4}
                                                    uint8 tensor = UTF-8 JSON of the non-tensor state
                                                    (quant_type, blocksize, dtype, shape, nested_blocksize,
                                                    nested_dtype, nested_offset)

  value[i] = quant_map[code[i]] * scale[i // blocksize]
  scale    = absmax                                                     (single quantisation)
           = dequant_blockwise_8bit(absmax, nested_quant_map, nested_absmax, nested_blocksize) + nested_offset
                                                                        (double quantisation)

Because the codebook travels with the checkpoint, no NF4/FP4 table is hard-coded here.
"""
from __future__ import annotations

import json
from typing import Sequence

import torch


def unpack_quant_state(blob: torch.Tensor) -> dict:
    """The `quant_state.bitsandbytes__*` tensor is the UTF-8 bytes of a JSON object."""
    return json.loads(bytes(blob.cpu().numpy()).decode("utf-8"))


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def dequant_blockwise_8bit(codes: torch.Tensor, code: torch.Tensor, absmax: torch.Tensor, blocksize: int) -> torch.Tensor:
    """out[i] = code[codes[i]] * absmax[i // blocksize], float32, flat."""
    flat = codes.reshape(-1).to(torch.int64)
    n = flat.numel()
    n_blocks = (n + blocksize - 1) // blocksize
    if absmax.numel() != n_blocks:
        raise ValueError(f"8-bit blockwise: {n} codes at blocksize {blocksize} need {n_blocks} scales, got {absmax.numel()}")
    return code.float()[flat] * absmax.float().reshape(-1).repeat_interleave(blocksize)[:n]


def dequant_bnb4(
    weight: torch.Tensor,
    quant_map: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    shape: Sequence[int],
    nested_absmax: torch.Tensor | None = None,
    nested_quant_map: torch.Tensor | None = None,
    nested_blocksize: int | None = None,
    nested_offset: float | None = None,
) -> torch.Tensor:
    """Dequantize one bitsandbytes 4-bit weight to float32 of the given `shape`."""
    numel = _numel(shape)
    packed = weight.reshape(-1).to(torch.uint8)
    if packed.numel() != (numel + 1) // 2:
        raise ValueError(f"bnb4: {numel} elements need {(numel + 1) // 2} packed bytes, got {packed.numel()}")
    if quant_map.numel() != 16:
        raise ValueError(f"bnb4: codebook must have 16 entries, got {quant_map.numel()}")

    # first element in the HIGH nibble
    idx = torch.stack((packed >> 4, packed & 0x0F), dim=-1).reshape(-1)[:numel].to(torch.int64)

    if nested_absmax is not None:
        if nested_quant_map is None or nested_blocksize is None or nested_offset is None:
            raise ValueError("bnb4: double quantisation needs nested_quant_map, nested_blocksize and nested_offset")
        scale = dequant_blockwise_8bit(absmax, nested_quant_map, nested_absmax, int(nested_blocksize)) + float(nested_offset)
    else:
        scale = absmax.float().reshape(-1)

    n_blocks = (numel + blocksize - 1) // blocksize
    if scale.numel() != n_blocks:
        raise ValueError(f"bnb4: {numel} elements at blocksize {blocksize} need {n_blocks} scales, got {scale.numel()}")

    out = quant_map.float()[idx] * scale.repeat_interleave(blocksize)[:numel]
    return out.reshape(tuple(int(s) for s in shape))


def dequant_bnb4_checkpoint(
    weight: torch.Tensor,
    state_blob: torch.Tensor,
    absmax: torch.Tensor,
    quant_map: torch.Tensor,
    nested_absmax: torch.Tensor | None = None,
    nested_quant_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize from the tensors exactly as they appear in a HuggingFace bnb-4bit safetensors file."""
    st = unpack_quant_state(state_blob)
    if st.get("quant_type") not in ("nf4", "fp4"):
        raise ValueError(f"bnb4: unsupported quant_type {st.get('quant_type')!r}")
    if st.get("shape") is None:
        raise ValueError("bnb4: quant_state carries no shape")
    return dequant_bnb4(
        weight, quant_map, absmax, int(st["blocksize"]), st["shape"],
        nested_absmax=nested_absmax, nested_quant_map=nested_quant_map,
        nested_blocksize=st.get("nested_blocksize"), nested_offset=st.get("nested_offset"),
    )
