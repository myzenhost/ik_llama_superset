#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import logging
import argparse
import os
import sys
import json
import math
import re
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Sequence, SupportsIndex, cast
from transformers import AutoConfig, AutoTokenizer

import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf
from gguf.constants import GGUFValueType

# reuse model definitions from the conversion/ package
from conversion import LazyTorchTensor, ModelBase, get_model_class, ModelType, get_model_architecture

logger = logging.getLogger("lora-to-gguf")


@dataclass
class PartialLoraTensor:
    A: Tensor | None = None
    B: Tensor | None = None


# magic to support tensor shape modifications and splitting
class LoraTorchTensor:
    _lora_A: Tensor  # (n_rank, row_size)
    _lora_B: Tensor  # (col_size, n_rank)
    _rank: int

    def __init__(self, A: Tensor, B: Tensor):
        assert len(A.shape) == len(B.shape)
        assert A.shape[-2] == B.shape[-1]
        if A.dtype != B.dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        self._lora_A = A
        self._lora_B = B
        self._rank = B.shape[-1]

    def get_lora_A_B(self) -> tuple[Tensor, Tensor]:
        return (self._lora_A, self._lora_B)

    def __getitem__(
        self,
        indices: (
            SupportsIndex
            | slice
            | tuple[SupportsIndex | slice | Tensor, ...]  # TODO: add ellipsis in the type signature
        ),
    ) -> LoraTorchTensor:
        shape = self.shape
        if isinstance(indices, SupportsIndex):
            if len(shape) > 2:
                return LoraTorchTensor(self._lora_A[indices], self._lora_B[indices])
            else:
                raise NotImplementedError  # can't return a vector
        elif isinstance(indices, slice):
            if len(shape) > 2:
                return LoraTorchTensor(self._lora_A[indices], self._lora_B[indices])
            else:
                return LoraTorchTensor(self._lora_A, self._lora_B[indices])
        elif isinstance(indices, tuple):
            assert len(indices) > 0
            if indices[-1] is Ellipsis:
                return self[indices[:-1]]
            # expand ellipsis
            indices = tuple(
                u
                for v in (
                    (
                        (slice(None, None) for _ in range(len(indices) - 1))
                        if i is Ellipsis
                        else (i,)
                    )
                    for i in indices
                )
                for u in v
            )

            if len(indices) < len(shape):
                indices = (*indices, *(slice(None, None) for _ in range(len(indices), len(shape))))

            # TODO: make sure this is correct
            indices_A = (
                *(
                    (
                        j.__index__() % self._lora_A.shape[i]
                        if isinstance(j, SupportsIndex)
                        else slice(None, None)
                    )
                    for i, j in enumerate(indices[:-2])
                ),
                slice(None, None),
                indices[-1],
            )
            indices_B = indices[:-1]
            return LoraTorchTensor(self._lora_A[indices_A], self._lora_B[indices_B])
        else:
            raise NotImplementedError  # unknown indice type

    @property
    def dtype(self) -> torch.dtype:
        assert self._lora_A.dtype == self._lora_B.dtype
        return self._lora_A.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        assert len(self._lora_A.shape) == len(self._lora_B.shape)
        return (*self._lora_B.shape[:-1], self._lora_A.shape[-1])

    def size(self, dim=None):
        assert dim is None
        return self.shape

    def dim(self) -> int:
        return len(self.shape)

    def contiguous(self) -> LoraTorchTensor:
        return LoraTorchTensor(
            self._lora_A.contiguous(),
            self._lora_B.contiguous(),
        )

    def reshape(self, *shape: int | tuple[int, ...]) -> LoraTorchTensor:
        if isinstance(shape[0], tuple):
            new_shape: tuple[int, ...] = shape[0]
        else:
            new_shape = cast(tuple[int, ...], shape)
        orig_shape = self.shape
        if len(new_shape) < 2:
            raise NotImplementedError  # can't become a vector

        # expand -1 in the shape
        if any(dim == -1 for dim in new_shape):
            n_elems = prod(orig_shape)
            n_new_elems = prod(dim if dim != -1 else 1 for dim in new_shape)
            assert n_elems % n_new_elems == 0
            new_shape = (*(dim if dim != -1 else n_elems // n_new_elems for dim in new_shape),)

        if new_shape[-1] != orig_shape[-1]:
            raise NotImplementedError  # can't reshape the row size trivially

        shape_A = (*(1 for _ in new_shape[:-2]), self._rank, orig_shape[-1])
        shape_B = (*new_shape[:-1], self._rank)
        return LoraTorchTensor(
            self._lora_A.reshape(shape_A),
            self._lora_B.reshape(shape_B),
        )

    def reshape_as(self, other: Tensor) -> LoraTorchTensor:
        return self.reshape(*other.shape)

    def view(self, *size: int) -> LoraTorchTensor:
        return self.reshape(*size)

    def permute(self, *dims: int) -> LoraTorchTensor:
        shape = self.shape
        dims = tuple(dim - len(shape) if dim >= 0 else dim for dim in dims)
        if dims[-1] == -1:
            # TODO: support higher dimensional A shapes bigger than 1
            assert all(dim == 1 for dim in self._lora_A.shape[:-2])
            return LoraTorchTensor(self._lora_A, self._lora_B.permute(*dims))
        if len(shape) == 2 and dims[-1] == -2 and dims[-2] == -1:
            return LoraTorchTensor(self._lora_B.permute(*dims), self._lora_A.permute(*dims))
        if len(shape) >= 3 and dims[-1] == -2 and dims[-2] == -1 and list(dims[:-2]) == list(range(-len(shape), -2)):
            # batched transpose of the last two dims (e.g. an expert stack [E, out, in] -> [E, in, out]): (B A)^T = A^T B^T
            # per leading index. Used by gpt_oss modify_tensors on target_parameters carriers (row
            # S69-ADAPTERS-TARGET-PARAMETERS-MOE-EXPERT-LORA-2026-10-26-01).
            return LoraTorchTensor(self._lora_B.transpose(-1, -2), self._lora_A.transpose(-1, -2))
        else:
            # TODO: compose the above two
            raise NotImplementedError

    def transpose(self, dim0: int, dim1: int) -> LoraTorchTensor:
        shape = self.shape
        dims = [i for i in range(len(shape))]
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(*dims)

    def swapaxes(self, axis0: int, axis1: int) -> LoraTorchTensor:
        return self.transpose(axis0, axis1)

    def split(self, split_size: int | Sequence[int], dim: int = 0) -> tuple[LoraTorchTensor, ...]:
        shape = self.shape
        ndim = len(shape)
        if dim < 0:
            dim += ndim
        if dim == ndim - 1:
            A_chunks = self._lora_A.split(split_size, dim=-1)
            return tuple(LoraTorchTensor(a, self._lora_B) for a in A_chunks)
        elif dim == ndim - 2:
            B_chunks = self._lora_B.split(split_size, dim=-2)
            return tuple(LoraTorchTensor(self._lora_A, b) for b in B_chunks)
        else:
            B_chunks = self._lora_B.split(split_size, dim=dim)
            if self._lora_A.shape[dim] == 1:
                return tuple(LoraTorchTensor(self._lora_A, b) for b in B_chunks)
            A_chunks = self._lora_A.split(split_size, dim=dim)
            return tuple(LoraTorchTensor(a, b) for a, b in zip(A_chunks, B_chunks))

    def to(self, *args, **kwargs):
        return LoraTorchTensor(self._lora_A.to(*args, **kwargs), self._lora_B.to(*args, **kwargs))

    def __mul__(self, other) -> LoraTorchTensor:
        # Only output-side multiplication for now
        # W = B @ A, so M_out * W == (M_out * B) @ A
        if not isinstance(other, (int, float)) and other.shape and other.shape[-1] != 1:
            raise NotImplementedError
        return LoraTorchTensor(self._lora_A, self._lora_B * other)

    def __rmul__(self, other) -> LoraTorchTensor:
        return self * other

    @classmethod
    def __torch_function__(cls, func: Callable, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.permute:
            assert len(args)
            return type(args[0]).permute(*args, **kwargs)
        elif func is torch.reshape:
            assert len(args)
            return type(args[0]).reshape(*args, **kwargs)
        elif func is torch.stack:
            assert len(args)
            assert isinstance(args[0], Sequence)
            dim = kwargs.get("dim", 0)
            assert dim == 0
            return LoraTorchTensor(
                torch.stack([a._lora_A for a in args[0]], dim),
                torch.stack([b._lora_B for b in args[0]], dim),
            )
        elif func is torch.cat:
            assert len(args)
            assert isinstance(args[0], Sequence)
            dim = kwargs.get("dim", 0)
            assert dim == 0
            if len(args[0][0].shape) > 2:
                return LoraTorchTensor(
                    torch.cat([a._lora_A for a in args[0]], dim),
                    torch.cat([b._lora_B for b in args[0]], dim),
                )
            elif all(torch.equal(args[0][0]._lora_A, t._lora_A) for t in args[0][1:]):
                return LoraTorchTensor(
                    args[0][0]._lora_A,
                    torch.cat([b._lora_B for b in args[0]], dim),
                )
            else:
                raise NotImplementedError
        elif func is torch.split:
            assert len(args) and len(args) >= 2
            tensor, split_size = args[0], args[1]
            dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
            return tensor.split(split_size, dim=dim)
        else:
            raise NotImplementedError


def get_base_tensor_name(lora_tensor_name: str) -> str:
    base_name = lora_tensor_name.replace("base_model.model.", "")
    base_name = base_name.replace(".lora_A.weight", ".weight")
    base_name = base_name.replace(".lora_B.weight", ".weight")
    # models produced by mergekit-extract-lora have token embeddings in the adapter
    base_name = base_name.replace(".lora_embedding_A", ".weight")
    base_name = base_name.replace(".lora_embedding_B", ".weight")
    # DoRA magnitude (one value per output row of the base Linear), saved by PEFT as "<module>.lora_magnitude_vector"
    # (peft 0.17.1 utils/save_and_load.py:113-121 strips the ".weight" of the newer ModuleDict layout)
    if is_dora_magnitude(lora_tensor_name):
        base_name = re.sub(r"\.lora_magnitude_vector(\.weight)?$", ".weight", base_name)
    for suffix in LILY_SUFFIXES:
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)] + ".weight"
    # IA3 scaling vector, saved by PEFT as "<module>.ia3_l" (the adapter-name key of the ParameterDict is stripped on save)
    base_name = re.sub(r"\.ia3_l(\.weight)?$", ".weight", base_name)
    return base_name


def is_ia3_vector(lora_tensor_name: str) -> bool:
    return re.search(r"\.ia3_l(\.weight)?$", lora_tensor_name) is not None


def ia3_side(vec_shape: tuple[int, ...], name: str) -> str:
    """"in" or "out": PEFT stores the IA3 vector as (1, in_features) for a feedforward module (the INPUT is scaled) and as
    (out_features, 1) otherwise (the OUTPUT is scaled); peft a7e8ea4a1227 tuners/ia3/layer.py update_layer (vector shape), :172-201 (forward)."""
    if len(vec_shape) == 2 and vec_shape[0] == 1 and vec_shape[1] > 1:
        return "in"
    if len(vec_shape) == 2 and vec_shape[1] == 1 and vec_shape[0] > 1:
        return "out"
    logger.error(f"IA3 vector '{name}' has shape {tuple(vec_shape)}: expected (1, in_features) or (out_features, 1)")
    sys.exit(1)


LILY_SUFFIXES = (".lily_A.weight", ".lily_B.weight", ".lily_router.weight")


def lily_part(lora_tensor_name: str) -> str | None:
    # "A", "B" or "router" for a PEFT Lily tensor (peft a7e8ea4a1227 tuners/lily/layer.py:30: lily_A, lily_B, lily_router are
    # nn.Linear modules, saved as "<module>.lily_X.weight"), else None
    for part, suffix in zip(("A", "B", "router"), LILY_SUFFIXES):
        if lora_tensor_name.endswith(suffix):
            return part
    return None


def lily_carrier(A: Tensor, B: Tensor, num_B: int, scaling: float) -> tuple[Tensor, Tensor]:
    # Carry Lily's num_B up-projections through the base model's own modify_tensors (llama's rope q/k row permute, row splits) as a
    # LoraTorchTensor whose B has the base OUTPUT rows: B_c = (scaling * lily_B.weight).T, [n_out, num_B * r], column e*r + k =
    # expert e, rank k (lily_B.weight is [num_B * r, n_out], reshaped by PEFT to (num_B, r, n_out), layer.py:168). A_c repeats A
    # num_B times only so the carrier has the [n_out, n_in] shape row ops expect; B_c A_c is never used as a weight.
    # PEFT's `scaling` is a direct multiplier on the delta (layer.py:177), so it is folded into B here.
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    r = A.shape[0]
    if B.shape[0] != num_B * r:
        raise ValueError(f"lily_B has {B.shape[0]} rows, want num_B * r = {num_B} * {r}")
    return A.repeat(num_B, 1), (B * float(scaling)).T.contiguous()


def is_dora_magnitude(lora_tensor_name: str) -> bool:
    return re.search(r"\.lora_magnitude_vector(\.weight)?$", lora_tensor_name) is not None


def dora_carrier(A: Tensor, B: Tensor, m: Tensor, lazy: bool) -> tuple[Tensor, Tensor]:
    # Carry the DoRA magnitude through the base model's own modify_tensors (row permutations such as llama's rope q/k
    # permute, row splits of fused projections, expert stacking) as ONE EXTRA RANK COLUMN of B, paired with an all-zero
    # row of A: B' A' = B A exactly, and every row operation applied to B is applied to m. LoraModel.modify_tensors strips
    # the column back off. Scalar/row-vector rescales of W (LoraTorchTensor.__mul__) scale m by the same factor, which is
    # what d = m / ||W + s B A||_row needs (a NEGATIVE per-row factor would need |c|; no converter in conversion/ uses one).
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    m = m.to(torch.float32)
    zero_row = torch.zeros((*A.shape[:-2], 1, A.shape[-1]), dtype=torch.float32)
    if lazy:
        zero_row = LazyTorchTensor.from_eager(zero_row)
    m_col = torch.reshape(m, (*B.shape[:-1], 1))
    return torch.cat([A, zero_row], dim=-2), torch.cat([B, m_col], dim=-1)


def get_peft_module_key(lora_tensor_name: str) -> str:
    # The module name PEFT matches rank_pattern / alpha_pattern against (its `current_key`, a name from the base
    # model's named_modules()): the saved tensor name minus the "base_model.model." prefix and the lora suffix.
    key = lora_tensor_name.replace("base_model.model.", "", 1)
    for suffix in (".lora_A.weight", ".lora_B.weight", ".lora_embedding_A", ".lora_embedding_B"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def peft_pattern_value(patterns: dict[str, Any], key: str, default: Any) -> Any:
    # Same rule as PEFT get_pattern_key: the FIRST pattern key k (dict order) with re.match(rf"(.*\.)?({k})$", key)
    # wins; no match -> the global default.
    # SOURCE-READ: peft 0.17.1 peft/utils/other.py:1244-1252 (get_pattern_key) and peft/tuners/lora/model.py:201-204
    # (r/alpha lookup); github.com/huggingface/peft @ a7e8ea4a1227 same logic; vLLM vllm/lora/peft_helper.py:100-150
    # (_match_pattern). Apache-2.0; reimplemented from the spec, no code copied.
    for k in patterns:
        if re.match(rf"(.*\.)?({k})$", key):
            return patterns[k]
    return default


def peft_scale_fold_factor(peft_key: str, rank: int, lparams: dict[str, Any]) -> float:
    # The fork applies scale = adapter_scale * lora_alpha / rank per tensor (src/llama-adapter.h get_scale; rank =
    # lora_b->ne[0]; lora_alpha is ONE GGUF value for the whole adapter, or no scaling when it is 0). PEFT applies
    # lora_alpha_m / r_m, or lora_alpha_m / sqrt(r_m) with use_rslora, where alpha_m comes from alpha_pattern
    # (SOURCE-READ: peft 0.17.1 peft/tuners/lora/layer.py:221-224). Any difference is folded into lora_b here, so the
    # GGUF is correct for every loader that reads adapter.lora.alpha (old binaries, export-lora, upstream) with no
    # new loader code; vLLM does the same fold (scaling into lora_b). Returns 1.0 for a plain LoRA, so those
    # adapters stay bit-identical to the pre-fold converter.
    r_pat = lparams.get("rank_pattern") or {}
    r_cfg = int(peft_pattern_value(r_pat, peft_key, lparams.get("r", rank)))
    if r_cfg != rank:
        raise ValueError(f"{peft_key}: tensor rank {rank} disagrees with adapter_config r/rank_pattern ({r_cfg})")
    alpha_glob = float(lparams["lora_alpha"])
    alpha_m = float(peft_pattern_value(lparams.get("alpha_pattern") or {}, peft_key, alpha_glob))
    s_peft = alpha_m / math.sqrt(rank) if lparams.get("use_rslora", False) else alpha_m / rank
    s_fork = alpha_glob / rank if alpha_glob else 1.0
    return s_peft / s_fork


# ---------------------------------------------------------------------------------------------------------------------------
# peft_type / variant dispatcher (row S69-ADAPTERS-CONVERTER-DISPATCHER-2026-10-26-01)
#
# Every adapter is either (a) rewritten into an EXACT low-rank pair (A, B) with B @ A == PEFT's own delta weight, or (b) refused
# loudly: exit code EXIT_REFUSED, and the message names the peft_type / flag and the row that would serve it. Nothing goes
# through PEFT convert_to_lora or any SVD: no truncation happens, so no numeric-rank threshold is ever applied (the dispatcher
# row's "SVD only at numeric rank" rule is met by never using SVD). DoRA keeps its own kind (dora_carrier) and is never
# flattened into a LoRA.
# SOURCE-READ (Apache-2.0, github.com/huggingface/peft @ a7e8ea4a12275b9b5846eb3e899e75ea6fdeeee1; reimplemented from the math,
# no code copied): tuners/adalora/layer.py:49-76,168-196 (SVDLinear: delta = B (A * E) * scaling / (ranknum + 1e-5), scaling =
# lora_alpha or r, ranknum = r); tuners/vera/layer.py:214-285 (delta = (lambda_b * B[:out]) @ (lambda_d * A[:, :in]));
# tuners/loha/layer.py:183-219,437-438 (delta = (w1a @ w1b) * (w2a @ w2b) * alpha / r); tuners/fourierft/layer.py:57-101 (delta =
# Re(ifft2(sparse spectrum)) * scaling, positions from randperm(random_loc_seed)); tuners/lokr/layer.py:256-287,497-503
# (delta = kron(w1, w2) * alpha / r, w1 = lokr_w1 or w1_a @ w1_b); tuners/lora/variants.py:489-582 (QA-LoRA forward: group-mean
# pool x over qalora_group_size, times in/group, then B A); tuners/lora/layer.py:2453-2693 (ParamWrapper: lora_A [r*E, in],
# lora_B [out, r*E], get_delta_factors), tuners/tuners_utils.py:1218-1311 (_inject_parameters nesting order).
# ---------------------------------------------------------------------------------------------------------------------------

EXIT_REFUSED = 3

ROW_DISPATCHER = "S69-ADAPTERS-CONVERTER-DISPATCHER-2026-10-26-01"
ROW_TARGET_PARAMETERS = "S69-ADAPTERS-TARGET-PARAMETERS-MOE-EXPERT-LORA-2026-10-26-01"
ROW_DENSE_DELTA = "S69-ADAPTERS-DENSE-DELTA-KIND-2026-10-26-01"
ROW_BIAS_DELTA = "S69-ADAPTERS-LORA-BIAS-BIAS-DELTA-KIND-2026-10-26-01"
ROW_VEC_SCALE = "S69-ADAPTERS-LINEAR-WRAPPER-IA3-RED-VEC-SCALE-2026-10-26-01"
ROW_ROTATION = "S69-ADAPTERS-ROTATION-OFT-BOFT-ROAD-2026-10-26-01"
ROW_INPUT_SIDE = "S69-ADAPTERS-INPUT-SIDE-PROMPT-PREFIX-CARTRIDGE-2026-10-26-01"
ROW_NORM_DELTA = "S69-ADAPTERS-NORM-DELTA-APPLY-2026-10-26-01"
ROW_ATTN_PROMPT = "S69-ADAPTERS-LLAMA-ADAPTER-ATTN-PROMPT-2026-10-26-01"
ROW_ROUTED = "S69-ADAPTERS-ROUTED-LORA-MUL-MAT-ID-STACK-2026-10-26-01"
# The remaining-low-rank row (S69-ADAPTERS-W3-CONVERTER-REMAINING-LOWRANK-TUNERS-2026-10-27-01) closed with every listed tuner
# dispatched; the variants still refused below (projection regeneration, UniLoRA all-linear, TinyLoRA non-Linear, WaveFT taps,
# PVeRA sampling) are tracked on their own row.
ROW_RESIDUAL_VARIANTS = "S69-ADAPTERS-W4-CONVERTER-RESIDUAL-VARIANTS-2026-10-27-01"

# peft_type -> (row, why there is no served kind). Every PeftType of peft a7e8ea4a1227 utils/peft_types.py:60-105 that is not
# dispatched below is listed here or falls to the unclassified refusal.
PEFT_TYPE_REFUSALS: dict[str, tuple[str, str]] = {
    "BEFT": (ROW_BIAS_DELTA, "BEFT trains biases (a BIAS_DELTA kind)"),
    "LN_TUNING": (ROW_NORM_DELTA, "LN_TUNING replaces norm weights (a NORM_DELTA kind)"),
    "OFT": (ROW_ROTATION, "OFT is a multiplicative input rotation (a ROTATION kind)"),
    "BOFT": (ROW_ROTATION, "BOFT is a multiplicative butterfly rotation (a ROTATION kind)"),
    "ROAD": (ROW_ROTATION, "RoAd is a multiplicative output rotation (a ROTATION kind)"),
    "HRA": (ROW_ROTATION, "HRA is a product of Householder reflections applied to W (a ROTATION kind)"),
    "PROMPT_TUNING": (ROW_INPUT_SIDE, "prompt tuning adds virtual input embeddings (an INPUT-SIDE kind)"),
    "MULTITASK_PROMPT_TUNING": (ROW_INPUT_SIDE, "multitask prompt tuning adds virtual input embeddings (an INPUT-SIDE kind)"),
    "P_TUNING": (ROW_INPUT_SIDE, "P-tuning adds encoder-produced virtual input embeddings (an INPUT-SIDE kind)"),
    "PREFIX_TUNING": (ROW_INPUT_SIDE, "prefix tuning adds per-layer KV prefixes (an INPUT-SIDE kind)"),
    "CPT": (ROW_INPUT_SIDE, "CPT adds context-aware prompt embeddings (an INPUT-SIDE kind)"),
    "CARTRIDGE": (ROW_INPUT_SIDE, "a Cartridge is a trained KV prefix (an INPUT-SIDE kind)"),
    "ADAPTION_PROMPT": (ROW_ATTN_PROMPT, "LLaMA-Adapter adds gated prompt attention (an ATTN_PROMPT kind)"),
    "XLORA": (ROW_ROUTED, "X-LoRA mixes LoRA experts with a learned per-token classifier (a ROUTED_LORA kind)"),
    "POLY": (ROW_ROUTED, "Poly routes over a LoRA skill inventory (a ROUTED_LORA kind)"),
    "C3A": (ROW_DENSE_DELTA, "C3A's delta is a block-circulant matrix: full rank, no exact low-rank form"),
    "SHIRA": (ROW_DENSE_DELTA, "SHiRA's delta is a sparse full-shape matrix"),
    "HIRA": (ROW_DENSE_DELTA, "HiRA's delta is W (Hadamard) B A: full rank and base-dependent"),
    "OSF": (ROW_DENSE_DELTA, "OSF fine-tunes inside an orthogonal subspace of W itself: the result is a rewritten dense W"),
}
# dispatched to an exact low-rank pair by lowrank_tuner_factors() (the second line: row S69-ADAPTERS-W3-CONVERTER-REMAINING-LOWRANK
# -TUNERS-2026-10-27-01; variants of these with no exact served form are refused inside lowrank_tuner_factors, naming the variant)
LOWRANK_PEFT_TYPES = ("ADALORA", "VERA", "LOHA", "LOKR", "FOURIERFT",
                      "VBLORA", "UNILORA", "TINYLORA", "DELORA", "GRALORA", "PSOFT", "PACA", "MISS", "RANDLORA", "PVERA", "WAVEFT")
# tuners whose rewrite needs the base weight SHAPES (read from the safetensors headers in --base)
NEEDS_BASE_SHAPES = ("VERA", "FOURIERFT", "UNILORA", "PACA", "MISS", "RANDLORA", "PVERA", "WAVEFT")
# LoRA init schemes that REWRITE the base weight at init (W_res = W - s B0 A0, or a requantised W): the saved A/B are only valid
# against that rewritten base. SOURCE-READ peft a7e8ea4a1227 tuners/lora/layer.py olora_init :411-456, pissa_init :458-491,
# corda_init :528-607, astra_init :612-676, loftq_init :681-694, lora_ga_init :713-826 (each assigns base_layer.weight.data).
# PEFT's save_pretrained(path_initial_model_for_weight_conversion=...) turns such an adapter into a plain one and writes
# init_lora_weights=True (peft_model.py:356-429), which this converter accepts.
BASE_REWRITING_INITS = ("pissa", "olora", "corda", "astra", "loftq", "lora_ga")


class PeftRefusal(Exception):
    pass


def refuse(what: str, row: str, why: str) -> None:
    raise PeftRefusal(f"REFUSED {what}: {why}. The fork has no served kind for it; tracked by row {row}. Nothing was written.")


def dispatch_peft_type(lparams: dict[str, Any]) -> str:
    """The handler for this adapter_config: "LORA", "LILY" or one of LOWRANK_PEFT_TYPES; raises PeftRefusal otherwise."""
    peft_type = str(lparams.get("peft_type") or "LORA").upper()
    if peft_type in PEFT_TYPE_REFUSALS:
        row, why = PEFT_TYPE_REFUSALS[peft_type]
        refuse(f"peft_type {peft_type}", row, why)
    if peft_type not in ("LORA", "LILY", "IA3", *LOWRANK_PEFT_TYPES):
        refuse(f"peft_type {peft_type}", ROW_DISPATCHER, "unclassified tuner: read its tuners/<name>/layer.py forward before mapping it")
    if lparams.get("modules_to_save"):
        refuse(f"peft_type {peft_type} with modules_to_save={lparams['modules_to_save']}", ROW_DENSE_DELTA,
               "modules_to_save stores full replacement weights for those modules (a DENSE_DELTA kind would serve them)")
    if lparams.get("trainable_token_indices"):
        refuse(f"peft_type {peft_type} with trainable_token_indices", ROW_DENSE_DELTA,
               "trainable tokens are dense replacement rows of the embedding (a DENSE_DELTA kind)")
    if str(lparams.get("bias") or "none") != "none":
        refuse(f"peft_type {peft_type} with bias={lparams.get('bias')!r}", ROW_BIAS_DELTA, "trained base biases are a BIAS_DELTA kind")
    if peft_type == "IA3" and lparams.get("fan_in_fan_out"):
        # a Conv1D base stores W transposed, so which side (in/out) a (1, n) / (n, 1) vector scales flips; not mapped
        refuse("IA3 with fan_in_fan_out", ROW_VEC_SCALE, "transposed (Conv1D) bases are not mapped for IA3")
    if peft_type in LOWRANK_PEFT_TYPES and peft_type != "TINYLORA" and lparams.get("fan_in_fan_out"):  # TINYLORA: Conv1D mapped (A/B stay (out,in))
        refuse(f"peft_type {peft_type} with fan_in_fan_out", ROW_DISPATCHER, "transposed (Conv1D) bases are not mapped for this tuner")
    if peft_type != "LORA":
        return peft_type
    # LoRA variants and flags
    if lparams.get("lora_bias"):
        refuse("LoRA lora_bias=True", ROW_BIAS_DELTA, "lora_B carries a trained bias (a BIAS_DELTA kind)")
    if lparams.get("layer_replication"):
        refuse("LoRA layer_replication", ROW_DISPATCHER, "the adapter targets base layers PEFT duplicated; the GGUF base has no such layers")
    if lparams.get("use_bdlora") and (lparams.get("use_dora") or lparams.get("use_qalora")):
        refuse("LoRA use_bdlora + use_dora/use_qalora", ROW_DISPATCHER, "PEFT's Linear variant table has no combined key "
               "(lora/layer.py:1034-1044; resolve_lora_variant :178-213 raises on it), so no such adapter exists to serve")
    if lparams.get("arrow_config"):
        refuse("LoRA arrow_config", ROW_ROUTED, "Arrow routes per token over a LoRA library (a ROUTED_LORA kind)")
    if lparams.get("kasa_config"):
        refuse("LoRA kasa_config (KaSA)", ROW_DISPATCHER, "KaSA truncates the base's own singular values (PEFT: supports_lora_conversion False)")
    if lparams.get("monteclora_config"):
        refuse("LoRA monteclora_config (MonteCLoRA)", ROW_DISPATCHER, "MonteCLoRA samples its factors; no deterministic served form is verified")
    if lparams.get("use_dora") and lparams.get("use_qalora"):
        refuse("LoRA use_dora + use_qalora", ROW_DISPATCHER, "PEFT itself does not implement the combination (lora/gptq.py:59-63)")
    init = lparams.get("init_lora_weights", True)
    if isinstance(init, str) and init.lower().startswith(BASE_REWRITING_INITS):
        refuse(f"LoRA init_lora_weights={init!r} (unconverted {init.lower().split('_')[0]})", ROW_DISPATCHER,
               "the adapter is relative to a base PEFT rewrote at init; re-save it with "
               "save_pretrained(path_initial_model_for_weight_conversion=...) to get a plain LoRA, then convert that")
    return "LORA"


def base_weight_shapes(dir_base_model: Path | None) -> dict[str, tuple[int, ...]]:
    """Tensor name -> shape of the base checkpoint, read from the safetensors headers only (no weights loaded). Empty when the base
    directory has no safetensors (e.g. a config-only hub cache dir)."""
    shapes: dict[str, tuple[int, ...]] = {}
    if dir_base_model is None or not Path(dir_base_model).is_dir():
        return shapes
    from safetensors import safe_open
    for f in sorted(Path(dir_base_model).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                shapes[k] = tuple(sf.get_slice(k).get_shape())
    return shapes


def base_in_features(shapes: dict[str, tuple[int, ...]], module_key: str) -> int | None:
    shp = shapes.get(module_key + ".weight")
    if shp is None:  # tolerate a differently prefixed checkpoint, but only on a unique suffix match
        hits = [s for k, s in shapes.items() if k.endswith("." + module_key + ".weight")]
        shp = hits[0] if len(hits) == 1 else None
    return int(shp[-1]) if shp is not None and len(shp) == 2 else None


def base_in_features_any(shapes: dict[str, tuple[int, ...]], module_key: str) -> int | None:
    # a dense base (".weight" [out, in]) or a GPTQ base (".g_idx" has one entry per input feature)
    n_in = base_in_features(shapes, module_key)
    if n_in is None:
        gi = shapes.get(module_key + ".g_idx")
        n_in = int(gi[0]) if gi is not None and len(gi) == 1 else None
    return n_in


def qalora_expand_state(lora_model: dict[str, Tensor], lparams: dict[str, Any], shapes: dict[str, tuple[int, ...]]) -> dict[str, Tensor]:
    """use_qalora=True: PEFT applies the QA-LoRA variant only on GPTQ layers (lora/gptq.py dispatch); on any other layer it is a
    plain LoRA with a full-width lora_A. Which one each module got is decided from the saved lora_A width against the base
    in_features: in/group -> expand exactly (qalora_expand_A), in -> plain, anything else (or no base shapes) -> refuse."""
    g = int(lparams.get("qalora_group_size", 16))
    out: dict[str, Tensor] = {}
    n_exp = n_plain = 0
    for name, t in lora_model.items():
        if name.endswith(".lora_A.weight"):
            key = get_peft_module_key(name)
            n_in = base_in_features_any(shapes, key)
            if n_in is None:
                raise PeftRefusal(f"REFUSED LoRA use_qalora on '{key}': the base in_features is needed to tell a pooled QA-LoRA lora_A "
                                  f"from a plain one and the base safetensors in --base do not have it; row {ROW_DISPATCHER}.")
            if t.shape[-1] == n_in:
                n_plain += 1
            elif t.shape[-1] * g == n_in:
                t = qalora_expand_A(t, g, n_in)
                n_exp += 1
            else:
                raise ValueError(f"QA-LoRA '{name}' {tuple(t.shape)}: width is neither in_features {n_in} nor in_features / {g}")
        out[name] = t
    logger.info(f"use_qalora: {n_exp} lora_A expanded from pooled groups of {g} (exact), {n_plain} plain-dispatched (non-GPTQ layer)")
    return out


def qalora_expand_A(A: Tensor, group: int, in_features: int) -> Tensor:
    # QA-LoRA forward (peft variants.py:560-576): delta = s B A (P * mean_g(x)), P = in/g pooled features, mean over CONTIGUOUS groups
    # of g inputs. Per input i that is s B A[:, i // g] * (P / g) * x_i, so the exact dense-input A is A[:, i // g] * P / g, rank r.
    if in_features % group != 0 or A.shape[-1] * group != in_features:
        raise ValueError(f"QA-LoRA lora_A {tuple(A.shape)} does not pool in_features {in_features} by group {group}")
    pooled = in_features // group
    return A.to(torch.float32).repeat_interleave(group, dim=-1) * (pooled / group)


def _module_parts(lora_model: dict[str, Tensor], suffixes: Sequence[str]) -> dict[str, dict[str, Tensor]]:
    parts: dict[str, dict[str, Tensor]] = {}
    for name, t in lora_model.items():
        key = name.replace("base_model.model.", "", 1)
        for suf in suffixes:
            if key.endswith("." + suf):
                parts.setdefault(key[: -len(suf) - 1], {})[suf] = t
                break
        else:
            raise ValueError(f"'{name}': not a tensor this tuner saves (expected one of {list(suffixes)})")
    return parts


def _eye(n: int) -> Tensor:
    return torch.eye(n, dtype=torch.float64)


def _full_to_factors(W: Tensor) -> tuple[Tensor, Tensor]:
    # exact W = U V with the smaller inner dimension: (W, I_in) or (I_out, W)
    return (W, _eye(W.shape[1])) if W.shape[1] <= W.shape[0] else (_eye(W.shape[0]), W)


def _compact(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    # an exact pair whose inner dimension reached min(out, in): the dense pair (W, I) / (I, W) is exact and smaller. No SVD: a
    # truncated SVD is exact only at numeric rank, and the dense pair needs no rank decision at all.
    if A.shape[0] >= min(B.shape[0], A.shape[1]):
        U, V = _full_to_factors(B @ A)
        return V, U
    return A, B


def _meta_hf_model(dir_base_model: Path | None, hf_model_id: str | None, trust_remote_code: bool):
    """The base model's HF module tree built on the meta device (no weights, no memory): module registration order and types."""
    from transformers import AutoModelForCausalLM, AutoModel
    src = hf_model_id or str(dir_base_model)
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=trust_remote_code)
    with torch.device("meta"):
        for auto in (AutoModelForCausalLM, AutoModel):
            try:
                return auto.from_config(cfg, trust_remote_code=trust_remote_code)
            except (ValueError, KeyError) as e:  # FAILURES-SURFACED: no auto class for this config; the next one is tried, all-fail raises
                logger.debug(f"meta model: {auto.__name__}.from_config failed: {e}")
    raise ValueError(f"cannot build the HF model of {src} on the meta device to read its module order")


class TunerBase:
    """Lazy access to what some tuners' exact delta needs beyond the adapter file: base weight shapes (safetensors headers), fp32
    base weights (PSOFT recomputes its frozen A/B from the SVD of W) and the module registration order (TinyLoRA's per-layer
    index, Uni-LoRA's regenerated index stream)."""

    def __init__(self, dir_base_model: Path | None, hf_model_id: str | None = None, trust_remote_code: bool = False,
                 shapes: dict[str, tuple[int, ...]] | None = None):
        self.dir = Path(dir_base_model) if dir_base_model is not None else None
        self.hf_model_id = hf_model_id
        self.trust_remote_code = trust_remote_code
        self._shapes = shapes
        self._files: dict[str, Path] | None = None
        self._named: dict[str, Any] | None = None

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        if self._shapes is None:
            self._shapes = base_weight_shapes(self.dir)
        return self._shapes

    def shape(self, module_key: str) -> tuple[int, int] | None:
        shp = self.shapes.get(module_key + ".weight")
        if shp is None:
            hits = [s for k, s in self.shapes.items() if k.endswith("." + module_key + ".weight")]
            shp = hits[0] if len(hits) == 1 else None
        return (int(shp[0]), int(shp[1])) if shp is not None and len(shp) == 2 else None

    def weight(self, module_key: str) -> Tensor | None:
        from safetensors import safe_open
        if self._files is None:
            self._files = {}
            if self.dir is not None and self.dir.is_dir():
                for f in sorted(self.dir.glob("*.safetensors")):
                    with safe_open(str(f), framework="pt") as sf:
                        for k in sf.keys():
                            self._files[k] = f
        name = module_key + ".weight"
        if name not in self._files:
            hits = [k for k in self._files if k.endswith("." + name)]
            if len(hits) != 1:
                return None
            name = hits[0]
        with safe_open(str(self._files[name]), framework="pt") as sf:
            return sf.get_tensor(name)

    def named_modules(self) -> dict[str, Any]:
        if self._named is None:
            self._named = dict(_meta_hf_model(self.dir, self.hf_model_id, self.trust_remote_code).named_modules())
        return self._named

    def find(self, module_key: str) -> tuple[str, Any]:
        named = self.named_modules()
        if module_key in named:
            return module_key, named[module_key]
        hits = [k for k in named if k.endswith("." + module_key)]
        if len(hits) != 1:
            raise ValueError(f"module '{module_key}' matches {len(hits)} modules of the HF model")
        return hits[0], named[hits[0]]


def _is_linear_like(module: Any) -> bool:
    from transformers.pytorch_utils import Conv1D
    return isinstance(module, (torch.nn.Linear, Conv1D))


def _peft_targets_in_order(lparams: dict[str, Any], base: TunerBase, what: str) -> list[str]:
    """The Linear modules PEFT targets, in the base model's registration order, by PEFT's own matching rule (tuners_utils.py
    check_target_module_exists :2430-2480: exclude_modules first, then a str target is re.fullmatch'ed (utils/other.py:1592-1597),
    a list matches the key itself or any '.'-suffix). Used only where the adapter file does not list the modules itself."""
    tm = lparams.get("target_modules")
    named = base.named_modules()
    if tm is None:
        # PEFT fills None from TRANSFORMERS_MODELS_TO_*_TARGET_MODULES_MAPPING[model_type] (tuners_utils.py _prepare_adapter_config)
        refuse(f"{what} with target_modules=None", ROW_RESIDUAL_VARIANTS,
               "the per-architecture default target list lives in the PEFT version that saved the adapter and is not vendored; "
               "re-save with explicit target_modules (or all-linear)")
    if isinstance(tm, str) and tm.lower() == "all-linear":
        # _maybe_include_all_linear_layers (tuners_utils.py:2525-2585): the target becomes the SET of every Linear/Conv1D full name
        # except the output embedding module (PreTrainedModel.get_output_embeddings) or, for SEQ_CLS without one, the classifier
        # head. A key found in that set matches directly (:2467-2469), so layers_to_transform / layers_pattern never filter it.
        root = named.get("")
        oe = root.get_output_embeddings() if root is not None and hasattr(root, "get_output_embeddings") else None
        if oe is None and lparams.get("task_type") == "SEQ_CLS":
            refuse(f"{what} with all-linear on a SEQ_CLS adapter", ROW_RESIDUAL_VARIANTS,
                   "the classifier head to exclude is found by PEFT's SEQ_CLS_HEAD_NAMES and is not mapped here")
        head = next((k for k, m in named.items() if m is oe), None) if oe is not None else None
        tm = {k for k, m in named.items() if k and _is_linear_like(m) and k != head}
    ltr = lparams.get("layers_to_transform")
    lpat = lparams.get("layers_pattern")
    ex = lparams.get("exclude_modules")
    out = []
    for key, mod in named.items():
        if not key or not _is_linear_like(mod):
            continue
        if ex and (re.fullmatch(ex, key) if isinstance(ex, str) else (key in ex or any(key.endswith("." + e) for e in ex))):
            continue
        if isinstance(tm, str):
            hit = re.fullmatch(tm, key) is not None
        else:
            hit = key in tm
            if not hit:
                hit = any(key.endswith("." + t) for t in tm)
                using_idx = ltr is not None and (len(ltr) != 0 if isinstance(ltr, list) else True)
                if hit and using_idx:
                    # tuners_utils.py:2473-2505, the suffix branch only: the layer index is the integer after the layers_pattern
                    # name (or after the first parent when it is empty); with no index in the key the module is not matched
                    pats = [lpat] if isinstance(lpat, str) else list(lpat or [])
                    if not pats:
                        mm = re.match(r".*?\.[^.]*\.(?P<idx>\d+)\.", key)
                    else:
                        mm = None
                        for p in pats:
                            mm = re.match(rf"(?:^|.*?\.){p}\.(?P<idx>\d+)\.", key)
                            if mm is not None:
                                break
                    if mm is None:
                        hit = False
                    else:
                        idx = int(mm.group("idx"))
                        hit = (idx == ltr) if isinstance(ltr, int) else (idx in ltr)
        if hit:
            out.append(key)
    return out


def _wavelet_filters(family: str) -> tuple[list[float], list[float], int | None]:
    """(rec_lo, rec_hi, PEFT's WAVELET_REDUCTIONS row or None) of a WaveFT wavelet: PEFT's own table (peft/tuners/waveft/wavelet.py,
    the taps PEFT itself reconstructs with) or PyWavelets; refuses when neither is importable."""
    try:
        from peft.tuners.waveft.wavelet import Wavelet as _PeftWavelet
        from peft.tuners.waveft.constants import WAVELET_REDUCTIONS
        w = _PeftWavelet(family)
        red = WAVELET_REDUCTIONS.get(family)
        return list(w.rec_lo), list(w.rec_hi), (int(red[0]) if red is not None else None)
    except ImportError:
        pass
    try:
        import pywt  # ty: ignore[unresolved-import]
        w = pywt.Wavelet(family)
        return list(w.rec_lo), list(w.rec_hi), None
    except ImportError:
        refuse(f"peft_type WAVEFT (wavelet_family={family!r})", ROW_RESIDUAL_VARIANTS,
               "the wavelet filter taps come from peft.tuners.waveft or PyWavelets and neither is importable here; install the PEFT "
               "that saved the adapter (or pywavelets) and convert again")
    raise AssertionError("unreachable")


def _kaiming_uniform_gen(shape: tuple[int, ...], generator: Any, *, vera_style: bool = False, a: float = 0.0, dtype: Any = None) -> Tensor:
    """PEFT's seeded projection init. vera_style: vera/model.py:49-75 (fan_in, gain sqrt 2, uniform(-b, b), float32); else
    torch.nn.init.kaiming_uniform_(a) as pvera/model.py:116-117 (a=0) and randlora/model.py:38-64 (a=sqrt 5, bf16/fp16 tensor).
    Written from the formulas; the same Generator draw order gives the same numbers on the same torch build (PEFT warns it may
    differ across torch versions and platforms)."""
    import math
    t = torch.empty(shape, dtype=dtype or torch.float32)
    with torch.no_grad():
        if vera_style:
            fan = t.shape[1] * (t[0][0].numel() if t.dim() > 2 else 1)
            bound = math.sqrt(3.0) * math.sqrt(2.0) / math.sqrt(fan)
            return t.uniform_(-bound, bound, generator=generator)
        return torch.nn.init.kaiming_uniform_(t, a=a, generator=generator)


def _max_dims(parts: dict[str, Any], need_shape: Any) -> tuple[int, int]:
    """_find_dim: elementwise max of (out, in) over the adapted modules (the modules the adapter file lists)."""
    shp = [need_shape(k) for k in parts]
    if not shp:
        raise ValueError("projection regeneration: the adapter file lists no adapted module")
    return max(s[0] for s in shp), max(s[1] for s in shp)


def _regen_shared_projection(kind: str, lparams: dict[str, Any], dims: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """The shared projections of VERA / PVERA / RANDLORA when save_projection=False, regenerated from projection_prng_key exactly
    as the tuner's _init_*_A_*_B does at load. dims = elementwise max (out, in) over the targeted modules (_find_dim)."""
    import math
    out_max, in_max = dims
    r = int(lparams.get("r", 8 if kind == "VERA" else 32 if kind == "RANDLORA" else 4))
    gen = torch.Generator(device="cpu").manual_seed(int(lparams.get("projection_prng_key", 0)))
    if kind == "VERA":
        A = _kaiming_uniform_gen((r, in_max), gen, vera_style=True)
        B = _kaiming_uniform_gen((out_max, r), gen, vera_style=True)
    elif kind == "PVERA":
        A = _kaiming_uniform_gen((r * 2, in_max), gen)
        B = _kaiming_uniform_gen((out_max, r), gen)
    else:  # RANDLORA (randlora/model.py:138-215)
        mx, mn = max(out_max, in_max), min(out_max, in_max)
        nb = -(-mn // r)
        sparsity = math.sqrt(mn) if lparams.get("very_sparse") else 3 if lparams.get("sparse") else None
        if sparsity is None:
            try:
                from accelerate.utils.imports import is_bf16_available
                dt = torch.bfloat16 if is_bf16_available() else torch.float16
            except ImportError:
                dt = torch.bfloat16
            A = _kaiming_uniform_gen((r, 1, mn), gen, a=math.sqrt(5), dtype=dt)
            B = torch.cat([_kaiming_uniform_gen((mx, 1, r), gen, a=math.sqrt(5), dtype=dt) for _ in range(nb)], dim=1)
        else:
            A0 = torch.rand((r, 1, mn), generator=gen)
            B0 = torch.rand((mx, nb, r), generator=gen)
            A, B = torch.zeros(A0.shape), torch.zeros(B0.shape)
            B[B0 < 1 / (2 * sparsity)] = -1
            B[B0 > 1 - 1 / (2 * sparsity)] = 1
            A[A0 < 1 / (2 * sparsity)] = -1
            A[A0 > 1 - 1 / (2 * sparsity)] = 1
        A, B = A / A.std(), B / B.std()
    logger.warning(f"{kind} save_projection=False: shared projections regenerated from projection_prng_key="
                   f"{int(lparams.get('projection_prng_key', 0))} with this torch's RNG; PEFT warns the draw may differ across torch "
                   "versions/platforms, so verify against the training run if the adapter came from another torch build")
    return A, B


def lowrank_tuner_factors(peft_type: str, lora_model: dict[str, Tensor], lparams: dict[str, Any],
                          shapes: dict[str, tuple[int, ...]], base: TunerBase | None = None) -> dict[str, tuple[Tensor, Tensor]]:
    """PEFT module key -> (A [R, in], B [out, R]) with B @ A EXACTLY the tuner's delta weight, every scale folded into B (the GGUF
    then carries lora_alpha 0, i.e. the loader's scale is the user scale). Computed in float64, returned float32."""
    out: dict[str, tuple[Tensor, Tensor]] = {}
    if base is None:
        base = TunerBase(None, shapes=shapes)

    def need_shape(key: str) -> tuple[int, int]:
        shp = base.shape(key)
        if shp is None:
            raise PeftRefusal(f"REFUSED peft_type {peft_type}: the [out, in] shape of '{key}' is needed and the base safetensors in "
                              f"--base do not have it; row {ROW_DISPATCHER}.")
        return shp

    def shared(*names: str, why: str) -> list[Tensor]:
        got = [lora_model.get(n) for n in names]
        if any(g is None for g in got):
            refuse(f"peft_type {peft_type} with save_projection=False", ROW_RESIDUAL_VARIANTS,
                   f"{', '.join(names)} are not in the file; {why}. Re-save with save_projection=True")
        return [g.to(torch.float64) for g in got]  # ty: ignore[possibly-unbound-attribute]

    def without(*names: str) -> dict[str, Tensor]:
        return {k: v for k, v in lora_model.items() if k not in names and not k.startswith(names)}

    def r_alpha(key: str, r_shape: int | None, alpha_name: str) -> tuple[int, float]:
        r_cfg = int(peft_pattern_value(lparams.get("rank_pattern") or {}, key, lparams.get("r", r_shape)))
        if r_shape is not None and r_cfg != r_shape:
            raise ValueError(f"{key}: tensor rank {r_shape} disagrees with adapter_config r/rank_pattern ({r_cfg})")
        alpha = float(peft_pattern_value(lparams.get("alpha_pattern") or {}, key, lparams.get(alpha_name, r_cfg)))
        return r_cfg, alpha

    if peft_type == "ADALORA":
        for key, p in _module_parts(lora_model, ("lora_A", "lora_B", "lora_E")).items():
            if set(p) != {"lora_A", "lora_B", "lora_E"}:
                raise ValueError(f"AdaLoRA '{key}': have {sorted(p)}, need lora_A, lora_B, lora_E")
            A, B, E = (p[k].to(torch.float64) for k in ("lora_A", "lora_B", "lora_E"))
            r = A.shape[0]  # ranknum = the loaded rank (a pruned adapter is reloaded at its kept rank, adalora/model.py resize)
            alpha = float(lparams.get("lora_alpha", 0) or 0)
            scaling = alpha if alpha > 0 else float(r)
            out[key] = (A * E, B * (scaling / (float(r) + 1e-5)))
    elif peft_type == "VERA":
        shared_A = lora_model.get("base_model.vera_A")
        shared_B = lora_model.get("base_model.vera_B")
        rest = {k: v for k, v in lora_model.items() if k not in ("base_model.vera_A", "base_model.vera_B")}
        if shared_A is None or shared_B is None:
            shared_A, shared_B = _regen_shared_projection("VERA", lparams, _max_dims(_module_parts(rest, ("vera_lambda_b", "vera_lambda_d")), need_shape))
        for key, p in _module_parts(rest, ("vera_lambda_b", "vera_lambda_d")).items():
            lb, ld = p["vera_lambda_b"].to(torch.float64), p["vera_lambda_d"].to(torch.float64)
            n_in = base_in_features(shapes, key)
            if n_in is None:
                raise PeftRefusal(f"REFUSED peft_type VERA: in_features of '{key}' is needed to slice the shared vera_A and the base "
                                  f"weights' safetensors are not in --base; row {ROW_DISPATCHER}.")
            A_s = shared_A.to(torch.float64)[: ld.shape[0], :n_in]
            B_s = shared_B.to(torch.float64)[: lb.shape[0], : ld.shape[0]]
            out[key] = (ld[:, None] * A_s, lb[:, None] * B_s)
    elif peft_type == "LOHA":
        for key, p in _module_parts(lora_model, ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2")).items():
            if "hada_t1" in p or "hada_t2" in p:
                raise PeftRefusal(f"REFUSED peft_type LOHA effective-conv (hada_t1/t2) on '{key}': a conv kernel, not a Linear; row {ROW_DISPATCHER}.")
            w1a, w1b, w2a, w2b = (p[k].to(torch.float64) for k in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"))
            r, alpha = r_alpha(key, int(w1a.shape[1]), "alpha")
            # Khatri-Rao: (w1a w1b) * (w2a w2b) = [w1a (row-wise kron) w2a] @ [w1b (column-wise kron) w2b], rank r1 * r2
            B = (w1a[:, :, None] * w2a[:, None, :]).reshape(w1a.shape[0], -1)
            A = (w1b[:, None, :] * w2b[None, :, :]).reshape(-1, w1b.shape[1])
            out[key] = (A, B * (alpha / r))
    elif peft_type == "LOKR":
        sufs = ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "lokr_t2")
        for key, p in _module_parts(lora_model, sufs).items():
            if "lokr_t2" in p:
                raise PeftRefusal(f"REFUSED peft_type LOKR effective-conv (lokr_t2) on '{key}': a conv kernel, not a Linear; row {ROW_DISPATCHER}.")
            f = {k: v.to(torch.float64) for k, v in p.items()}
            U1, V1 = _full_to_factors(f["lokr_w1"]) if "lokr_w1" in f else (f["lokr_w1_a"], f["lokr_w1_b"])
            U2, V2 = _full_to_factors(f["lokr_w2"]) if "lokr_w2" in f else (f["lokr_w2_a"], f["lokr_w2_b"])
            r_shape = int(f["lokr_w1_a"].shape[1]) if "lokr_w1_a" in f else (int(f["lokr_w2_a"].shape[1]) if "lokr_w2_a" in f else None)
            r, alpha = r_alpha(key, r_shape, "alpha")
            # kron(U1 V1, U2 V2) = kron(U1, U2) @ kron(V1, V2)
            out[key] = (torch.kron(V1, V2), torch.kron(U1, U2) * (alpha / r))
    elif peft_type == "FOURIERFT":
        # delta = Re(ifft2(D)) * scaling, D [out, in] zero except D[p_k, q_k] = spectrum[k] (fourierft/layer.py:94-101); the
        # positions are torch.randperm(out*in, Generator().manual_seed(random_loc_seed))[:n], row = idx // in, col = idx % in
        # (layer.py:72-78; not saved, PEFT regenerates them the same way at load). ifft2 (norm "backward") gives
        # delta[m, c] = scaling/(M N) * sum_k s_k Re(e^{i a_k m} e^{i b_k c}), a_k = 2 pi p_k / M, b_k = 2 pi q_k / N, and
        # Re(e^{ia} e^{ib}) = cos a cos b - sin a sin b: per distinct column frequency q one cos row and one sin row of A (the sin
        # row is identically 0 when q = 0 or 2q = N and is dropped) and the matching B columns summed over its k. Exact.
        seed = int(lparams.get("random_loc_seed", 777))
        scaling = float(lparams.get("scaling", 150.0))
        for key, p in _module_parts(lora_model, ("fourierft_spectrum",)).items():
            spec = p["fourierft_spectrum"].to(torch.float64)
            n = int(spec.shape[0])
            n_cfg = int(peft_pattern_value(lparams.get("n_frequency_pattern") or {}, key, lparams.get("n_frequency", n)))
            if n_cfg != n:
                raise ValueError(f"FourierFT '{key}': spectrum has {n} entries, adapter_config n_frequency/pattern says {n_cfg}")
            shp = shapes.get(key + ".weight")
            if shp is None or len(shp) != 2:
                raise PeftRefusal(f"REFUSED peft_type FOURIERFT: the [out, in] shape of '{key}' is needed to regenerate the spectrum "
                                  f"positions and the base safetensors in --base do not have it; row {ROW_DISPATCHER}.")
            M, N = int(shp[0]), int(shp[1])
            idx = torch.randperm(M * N, generator=torch.Generator().manual_seed(seed))[:n]
            rows, cols = idx // N, idx % N
            coef = spec * (scaling / (M * N))
            m_ax = torch.arange(M, dtype=torch.float64)
            c_ax = torch.arange(N, dtype=torch.float64)
            a_rows: list[Tensor] = []
            b_cols: list[Tensor] = []
            for q in torch.unique(cols).tolist():
                sel = cols == q
                ang_m = 2.0 * math.pi * torch.outer(m_ax, rows[sel].to(torch.float64)) / M  # [M, k_q]
                beta = 2.0 * math.pi * q * c_ax / N
                a_rows.append(torch.cos(beta))
                b_cols.append((torch.cos(ang_m) * coef[sel]).sum(1))
                if q != 0 and 2 * q != N:
                    a_rows.append(torch.sin(beta))
                    b_cols.append(-(torch.sin(ang_m) * coef[sel]).sum(1))
            A = torch.stack(a_rows)
            B = torch.stack(b_cols, dim=1)
            if A.shape[0] >= min(M, N):  # more frequencies than the smaller side: the dense factorisation is the smaller exact one
                U, V = _full_to_factors(B @ A)
                A, B = V, U
            out[key] = (A, B)
    # ---- row S69-ADAPTERS-W3-CONVERTER-REMAINING-LOWRANK-TUNERS-2026-10-27-01. SOURCE-READ peft a7e8ea4a1227 (Apache-2.0), each
    # branch cites the forward / get_delta_weight it reproduces; the math is rewritten here, no code copied.
    elif peft_type == "VBLORA":
        # A = sum_k softmax(topk logits)_k bank[idx_k] over each (row, tile), reshaped (r, in); B likewise over logits_B
        # [out/L, r, nv], transposed to (out, r); delta = B @ A, no scale (vblora/layer.py:182-215,217-233). save_only_topk_weights
        # saves the first topk-1 softmax weights and the indices; PEFT rebuilds the last as 1 - sum (vblora/model.py:244-270).
        bank = lora_model["base_model.vblora_vector_bank"].to(torch.float64)
        topk = int(lparams.get("topk", 2))
        sufs = ("vblora_logits_A", "vblora_logits_B", "vblora_logits_A_topk_indices", "vblora_logits_A_topk_weights",
                "vblora_logits_B_topk_indices", "vblora_logits_B_topk_weights")

        def compose(p: dict[str, Tensor], side: str) -> Tensor:
            if f"vblora_logits_{side}" in p:
                top, idx = p[f"vblora_logits_{side}"].topk(topk, dim=-1)
                w = torch.softmax(top.to(torch.float64), dim=-1)
            else:
                idx = p[f"vblora_logits_{side}_topk_indices"].long()
                w = p[f"vblora_logits_{side}_topk_weights"].to(torch.float64)
                w = torch.cat([w, 1.0 - w.sum(-1, keepdim=True)], dim=-1)
            return (w.unsqueeze(-1) * bank[idx.long()]).sum(-2)

        for key, p in _module_parts(without("base_model.vblora_vector_bank"), sufs).items():
            a3 = compose(p, "A")  # [r, in/L, L]
            b3 = compose(p, "B")  # [out/L, r, L]
            out[key] = (a3.reshape(a3.shape[0], -1), b3.transpose(1, 2).reshape(-1, b3.shape[1]))
    elif peft_type == "UNILORA":
        # A = theta[idx_A] * scale_A, B = theta[idx_B] * scale_B, delta = B @ A (unilora/layer.py:204-230). With save_indices=False
        # the file holds theta only and PEFT regenerates the indices at load (unilora/model.py:83-151): one balanced index stream
        # for all Linear layers in model.modules() order (each index floor/ceil(D/d) times, numpy default_rng(proj_seed) choice of
        # the remainder then shuffle), chunked A then B per layer, scales = 1/sqrt(count) in float32.
        theta = lora_model["base_model.unilora_theta_d"]
        rest = without("base_model.unilora_theta_d")
        if rest:
            parts = _module_parts(rest, ("unilora_indices_A", "unilora_indices_B", "unilora_scales_A", "unilora_scales_B"))
        else:
            import numpy as np
            r = int(lparams.get("r", 4))
            d = int(lparams.get("theta_d_length", theta.shape[0]))
            if d != theta.shape[0]:
                raise ValueError(f"Uni-LoRA theta_d has {theta.shape[0]} entries, adapter_config theta_d_length {d}")
            layers = [(k, need_shape(k)) for k in _peft_targets_in_order(lparams, base, "peft_type UNILORA save_indices=False")]
            if not layers:
                raise ValueError("Uni-LoRA save_indices=False: target_modules matched no Linear module of the base model")
            total = sum(r * n_in + n_out * r for _, (n_out, n_in) in layers)
            rng = np.random.default_rng(int(lparams.get("proj_seed", 42)))
            data = np.repeat(np.arange(d), total // d)
            if total % d:
                data = np.concatenate([data, rng.choice(d, size=total % d, replace=False)])
            rng.shuffle(data)
            stream = torch.as_tensor(data, dtype=torch.long)
            counts = torch.bincount(stream, minlength=d)
            inv = torch.zeros(d, dtype=torch.float32)
            inv[counts > 0] = counts[counts > 0].float().rsqrt()
            parts, pos = {}, 0
            for k, (n_out, n_in) in layers:
                iA = stream[pos: pos + r * n_in].view(r, n_in)
                pos += r * n_in
                iB = stream[pos: pos + n_out * r].view(n_out, r)
                pos += n_out * r
                parts[k] = {"unilora_indices_A": iA, "unilora_indices_B": iB, "unilora_scales_A": inv[iA], "unilora_scales_B": inv[iB]}
            logger.info(f"UNILORA save_indices=False: index stream of {total} regenerated over {len(layers)} module(s)")
        th = theta.to(torch.float64)
        for key, p in parts.items():
            A = th[p["unilora_indices_A"].long()] * p["unilora_scales_A"].to(torch.float64)
            B = th[p["unilora_indices_B"].long()] * p["unilora_scales_B"].to(torch.float64)
            out[key] = (A, B)
    elif peft_type == "TINYLORA":
        # delta = B @ R @ A, R = sum_i v[i] P[i] (tinylora/layer.py:244-290). v is shared per group: num_groups = max(1,
        # round(N (1 - weight_tying))), group_size = max(1, N // num_groups), group = min(idx // group_size, num_groups - 1), idx =
        # the module's rank among the N targeted modules in named_modules() order (tinylora/model.py:90-170). P not saved
        # (save_projection=False) is regenerated as PEFT does: N(0, 1/sqrt(r)) of shape (u, r, r) from Generator().manual_seed(
        # projection_seed + idx) (layer.py:135-137,231-241).
        vs = {k.rsplit(".", 1)[1]: t.to(torch.float64) for k, t in lora_model.items() if k.startswith("base_model.tinylora_v.")}
        parts = _module_parts(without("base_model.tinylora_v."), ("tinylora_A", "tinylora_B", "tinylora_P"))
        names = list(base.named_modules())
        pos = {k: names.index(base.find(k)[0]) for k in parts}
        for k in parts:
            # Conv1D (GPT-2 style) is the Linear form: PEFT takes the SVD of weight.T for it, so A is (r, in), B is (out, r)
            # (tinylora/layer.py Linear, fan_in_fan_out) and delta = B R A holds unchanged.
            if not (isinstance(base.find(k)[1], torch.nn.Linear) or type(base.find(k)[1]).__name__ == "Conv1D"):
                refuse(f"peft_type TINYLORA on '{k}' ({type(base.find(k)[1]).__name__})", ROW_RESIDUAL_VARIANTS,
                       "TinyLoRA on nn.Linear / Conv1D is mapped; an Embedding's delta is a lookup-table delta B[x] R A "
                       "(tinylora/layer.py:598-603) with no (out,in) low-rank GGUF form")
        order = {k: i for i, k in enumerate(sorted(parts, key=lambda k: pos[k]))}
        n_mod = len(parts)
        wt = float(lparams.get("weight_tying", 0.0))
        num_groups = max(1, round(n_mod * (1.0 - wt)))
        group_size = max(1, n_mod // num_groups)
        seed = int(lparams.get("projection_seed", 42))
        for key, p in parts.items():
            A, B = p["tinylora_A"].to(torch.float64), p["tinylora_B"].to(torch.float64)
            r = A.shape[0]
            idx = order[key]
            g = str(min(idx // group_size, num_groups - 1))
            if g not in vs:
                raise ValueError(f"TinyLoRA '{key}': layer index {idx} -> group {g}, but the file has v groups {sorted(vs)}")
            v = vs[g]
            if "tinylora_P" in p:
                P = p["tinylora_P"].to(torch.float64)
            else:
                P = torch.normal(mean=0.0, std=1.0 / (r ** 0.5), size=(v.shape[0], r, r),
                                 generator=torch.Generator().manual_seed(seed + idx)).to(torch.float64)
            R = torch.einsum("i,ijk->jk", v, P)
            out[key] = (A, B @ R)
    elif peft_type == "DELORA":
        # delta = B diag(lambda / r / (max(||A_i||, 1e-4) max(||B^i||, 1e-4))) A, each input column times w_norm
        # (delora/layer.py:63-73; r = rank_pattern / r, delora/model.py:75-78)
        for key, p in _module_parts(lora_model, ("delora_A", "delora_B", "delora_lambda", "delora_w_norm")).items():
            A, B = p["delora_A"].to(torch.float64), p["delora_B"].to(torch.float64)
            lam, wn = p["delora_lambda"].to(torch.float64), p["delora_w_norm"].to(torch.float64)
            r, _ = r_alpha(key, int(A.shape[0]), "delora_lambda")
            an = torch.clamp(A.norm(dim=1), min=1e-4)
            bn = torch.clamp(B.norm(dim=0), min=1e-4)
            d = lam / r / (an * bn)
            out[key] = ((d[:, None] * A) * wn[None, :], B)
    elif peft_type == "GRALORA":
        # forward (gralora/layer.py:320-386): x is split into N = gralora_k input blocks, h_n = x_n A_n ([in/N, R]), regrouped
        # so output block j = sum_n h_n[j s:(j+1) s] B_j[n s:(n+1) s, :] with s = R / N. So delta block (j, n) = B_j[n s:(n+1) s]^T
        # A_n[:, j s:(j+1) s]^T: N*N rank-s pieces (rank N R), plus the hybrid B_general A_general; scaling alpha / (r + hybrid_r)
        # (layer.py:155).
        sufs = ("gralora_A", "gralora_B", "gralora_A_general.weight", "gralora_B_general.weight")
        for key, p in _module_parts(lora_model, sufs).items():
            A3, B3 = p["gralora_A"].to(torch.float64), p["gralora_B"].to(torch.float64)
            N, ib, R = A3.shape
            ob = B3.shape[2]
            if B3.shape[:2] != (N, R) or R % N:
                raise ValueError(f"GraLoRA '{key}': gralora_A {tuple(A3.shape)} / gralora_B {tuple(B3.shape)} are not a k x k grid")
            s = R // N
            n_in, n_out = N * ib, N * ob
            hyb = int(p["gralora_A_general.weight"].shape[0]) if "gralora_A_general.weight" in p else 0
            if hyb != int(lparams.get("hybrid_r", 0) or 0):
                raise ValueError(f"GraLoRA '{key}': hybrid rank {hyb} in the file, adapter_config hybrid_r {lparams.get('hybrid_r')}")
            scaling = float(lparams.get("alpha", 64)) / (int(lparams.get("r", R)) + hyb)
            a_rows, b_cols = [], []
            for j in range(N):
                for n in range(N):
                    a = torch.zeros(s, n_in, dtype=torch.float64)
                    a[:, n * ib:(n + 1) * ib] = A3[n][:, j * s:(j + 1) * s].T
                    b = torch.zeros(n_out, s, dtype=torch.float64)
                    b[j * ob:(j + 1) * ob, :] = B3[j][n * s:(n + 1) * s, :].T
                    a_rows.append(a)
                    b_cols.append(b)
            if hyb:
                a_rows.append(p["gralora_A_general.weight"].to(torch.float64))
                b_cols.append(p["gralora_B_general.weight"].to(torch.float64))
            out[key] = _compact(torch.cat(a_rows, 0), torch.cat(b_cols, 1) * scaling)
    elif peft_type == "PSOFT":
        # delta = scaling B0 (R - I) A0 (psoft/layer.py:369-395), scaling = psoft_alpha / r. A0/B0 are NOT saved: PEFT recomputes
        # them at load from the fp32 base W (layer.py:286-347): full SVD (torch.linalg.svd) or svd_lowrank(q=r, niter) under
        # fork_rng + manual_seed(random_seed); psoft_init A0 = Vh_r, B0 = U_r diag(S_r / scaling); pissa_init splits sqrt(S_r /
        # scaling) over both. R (OrthLayer.get_matrix :125-170): orth -> Q = skew(weight) and R = (I + Q)(I - Q)^-1 or its
        # Neumann series; else R = weight; then diag(vector_b) R diag(vector_a). The SVD is recomputed with the same torch call
        # as PEFT, so the result is exact against PEFT on the same torch/LAPACK (PEFT's own load has the same dependence).
        r = int(lparams.get("r", 32))
        scaling = float(lparams.get("psoft_alpha", 32)) / r
        sufs = ("psoft_R.weight", "psoft_R.vector_a", "psoft_R.vector_b")
        for key, p in _module_parts(lora_model, sufs).items():
            W = base.weight(key)
            if W is None:
                raise PeftRefusal(f"REFUSED peft_type PSOFT: '{key}' needs the base weight itself (PSOFT's A/B are the SVD of W, "
                                  f"recomputed at load) and the base safetensors in --base do not have it; row {ROW_DISPATCHER}.")
            W = W.to(torch.float32)
            if lparams.get("psoft_svd", "full") == "full":
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
            elif lparams.get("psoft_svd") == "lowrank":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(lparams.get("random_seed", 0) or 0))
                    U, S, V = torch.svd_lowrank(W, q=r, niter=int(lparams.get("psoft_svd_lowrank_niter", 10)))
                Ur, Sr, Vhr = U[:, :r], S[:r], V[:, :r].t()
            else:
                raise ValueError(f"PSOFT psoft_svd={lparams.get('psoft_svd')!r}")
            Ss = Sr / scaling
            init = lparams.get("ab_svd_init", "psoft_init")
            if init == "psoft_init":
                A0, B0 = Vhr, Ur @ torch.diag(Ss)
            elif init == "pissa_init":
                sq = torch.sqrt(Ss)
                A0, B0 = torch.diag(sq) @ Vhr, Ur @ torch.diag(sq)
            else:
                raise ValueError(f"PSOFT ab_svd_init={init!r}")
            w = p["psoft_R.weight"].to(torch.float64)
            eye = _eye(r)
            if lparams.get("psoft_orth", True):
                Q = torch.zeros(r, r, dtype=torch.float64)
                iu = torch.triu_indices(r, r, 1)
                Q[iu[0], iu[1]] = w
                Q = Q - Q.T
                if lparams.get("use_cayley_neumann", False):
                    eps = lparams.get("cayley_neumann_eps")
                    if eps is not None:
                        nq = torch.linalg.norm(Q, ord="fro")
                        if torch.isfinite(nq) and nq > float(eps):
                            Q = Q * (float(eps) / (nq + 1e-12))
                    t = int(lparams.get("num_cayley_neumann_terms", 5))
                    R = eye.clone()
                    if t > 1:
                        R = R + 2.0 * Q
                        if t > 2:
                            Qp = Q @ Q
                            R = R + 2.0 * Qp
                            for _ in range(3, t - 1):
                                Qp = Qp @ Q
                                R = R + 2.0 * Qp
                            R = R + Qp @ Q
                else:
                    R = torch.linalg.solve(eye - Q, eye + Q, left=False)
            else:
                R = w.reshape(r, r)
            if "psoft_R.vector_b" in p:
                R = p["psoft_R.vector_b"].to(torch.float64)[:, None] * R
            if "psoft_R.vector_a" in p:
                R = R * p["psoft_R.vector_a"].to(torch.float64)[None, :]
            out[key] = ((R - eye) @ A0.to(torch.float64), B0.to(torch.float64) * scaling)
    elif peft_type == "PACA":
        # delta is zero except the selected input columns: delta[:, idx] = paca_delta * paca_alpha / r (config r, also under
        # paca_indices_pattern; paca/layer.py:203-243,348-360). Exact rank-#columns pair: A = one-hot rows, B = the scaled columns.
        scaling = float(lparams.get("paca_alpha", 8)) / int(lparams.get("r", 8))
        for key, p in _module_parts(lora_model, ("paca_delta", "paca_indices")).items():
            n_out, n_in = need_shape(key)
            idx = p["paca_indices"].long()
            A = torch.zeros(idx.shape[0], n_in, dtype=torch.float64)
            A[torch.arange(idx.shape[0]), idx] = 1.0
            out[key] = (A, p["paca_delta"].to(torch.float64) * scaling)
    elif peft_type == "MISS":
        # standard / mini (miss/layer.py:275-326): delta[o, i] = block[i % r, o], the block [r, out] (mini: [r, mini_r] repeated
        # out / mini_r times along out, :300-301). So delta = block^T @ A with A[k, i] = [i % r == k]: exact, rank r. "bat" adds
        # W-dependent blocks (W_blk @ block + block, :234-273): base-dependent and full rank, so it is refused.
        init = lparams.get("init_weights", True)
        for key, p in _module_parts(lora_model, ("miss_block",)).items():
            blk = p["miss_block"].to(torch.float64)
            if init == "bat" or blk.dim() != 2:
                refuse("peft_type MISS init_weights='bat'", ROW_DENSE_DELTA,
                       "Bat's delta is W_blocks @ block + block: base-dependent and full rank (a DENSE_DELTA kind)")
            n_out, n_in = need_shape(key)
            if init == "mini":
                mini_r = int(lparams.get("mini_r", 1))
                if n_out % mini_r:
                    raise ValueError(f"MiSS mini '{key}': out_features {n_out} not divisible by mini_r {mini_r}")
                blk = blk.repeat(1, n_out // mini_r)
            r = blk.shape[0]
            if blk.shape[1] != n_out:
                raise ValueError(f"MiSS '{key}': block {tuple(blk.shape)} does not fit out_features {n_out}")
            A = torch.zeros(r, n_in, dtype=torch.float64)
            A[torch.arange(n_in) % r, torch.arange(n_in)] = 1.0
            out[key] = _compact(A, blk.T.contiguous())
    elif peft_type == "RANDLORA":
        # randlora/layer.py:262-325: with min/max of (out, in) and nb = #bases, update_A = (lambda[:, :, None] * A[:, :nb, :min] *
        # gamma[None]).flatten(0, 1) and update_B = B[:max, :nb, :].flatten(1); (first, second) = (update_A, update_B) when min
        # is in_features else (update_B^T, update_A^T); delta = scaling * second @ first, scaling = randlora_alpha / r (:128).
        # nb = ceil(min / r) makes the natural rank nb r >= min: the dense exact pair is used.
        rest = without("base_model.randlora_A", "base_model.randlora_B")
        if "base_model.randlora_A" in lora_model and "base_model.randlora_B" in lora_model:
            sA, sB = shared("base_model.randlora_A", "base_model.randlora_B", why="")
        else:
            sA, sB = (t.to(torch.float64) for t in _regen_shared_projection(
                "RANDLORA", lparams, _max_dims(_module_parts(rest, ("randlora_lambda", "randlora_gamma")), need_shape)))
        scaling = float(lparams.get("randlora_alpha", 640)) / int(lparams.get("r", 32))
        for key, p in _module_parts(rest, ("randlora_lambda", "randlora_gamma")).items():
            n_out, n_in = need_shape(key)
            lam, gam = p["randlora_lambda"].to(torch.float64), p["randlora_gamma"].to(torch.float64)
            nb = lam.shape[1]
            mn, mx = min(n_out, n_in), max(n_out, n_in)
            uB = sB[:mx, :nb, :].flatten(1)
            uA = (lam[:, :, None] * sA[:, :nb, :mn] * gam[None]).flatten(0, 1)
            first, second = (uA, uB) if mn == n_in else (uB.T, uA.T)
            out[key] = _compact(first, second * scaling)
    elif peft_type == "PVERA":
        # eval (sample_at_inference=False) uses the mean half only: delta = (lambda_b * B[:out]) @ (lambda_d[:h] * A[:h, :in]),
        # h = half the rows (pvera/layer.py:243-283; forward :316-317 takes mu = the first chunk and _reparametrize :159-170 returns
        # mu at eval). sample_at_inference=True draws noise per call: no deterministic served form, refused.
        if lparams.get("sample_at_inference"):
            refuse("peft_type PVERA with sample_at_inference=True", ROW_RESIDUAL_VARIANTS,
                   "the adapter samples mu + eps * std at every inference call; a fixed weight delta cannot serve it")
        rest = without("base_model.pvera_A", "base_model.pvera_B")
        if "base_model.pvera_A" in lora_model and "base_model.pvera_B" in lora_model:
            sA, sB = shared("base_model.pvera_A", "base_model.pvera_B", why="")
        else:
            sA, sB = (t.to(torch.float64) for t in _regen_shared_projection(
                "PVERA", lparams, _max_dims(_module_parts(rest, ("pvera_lambda_b", "pvera_lambda_d")), need_shape)))
        for key, p in _module_parts(rest, ("pvera_lambda_b", "pvera_lambda_d")).items():
            n_out, n_in = need_shape(key)
            lb, ld = p["pvera_lambda_b"].to(torch.float64), p["pvera_lambda_d"].to(torch.float64)
            A_s = sA[:, :n_in]
            h = A_s.shape[0] // 2
            if ld.shape[0] // 2 != h or sB.shape[1] != h or lb.shape[0] != n_out:
                raise ValueError(f"PVeRA '{key}': lambda_d {tuple(ld.shape)} / lambda_b {tuple(lb.shape)} / pvera_A {tuple(sA.shape)} "
                                 f"/ pvera_B {tuple(sB.shape)} do not fit")
            out[key] = (ld[:h, None] * A_s[:h], lb[:, None] * sB[:n_out, :])
    elif peft_type == "WAVEFT":
        # waveft/layer.py:124-197. Positions: randperm(out*in, Generator().manual_seed(random_loc_seed))[:n] -> (idx // in, idx % in)
        # (:95-101). use_idwt=False: delta = sparse * scaling. use_idwt=True: the spectrum is placed (centred) in a zero matrix padded
        # by WAVELET_REDUCTIONS and made even, split into quadrants cA [:H2,:W2], cH [:H2,W2:], cV [H2:,:W2], cD [H2:,W2:], and one
        # level of waverec2d (waverec2d.py:257-315): conv_transpose2d stride 2 with filters outer(lo,lo), outer(hi,lo),
        # outer(lo,hi), outer(hi,hi) (rows x cols), crop (2L-3)//2 per side, times scaling, centre-crop to out x in. So each
        # coefficient c at band position (p, q) adds c * u v^T with u[2p+i] = row filter[i], v[2q+j] = col filter[j]: the row
        # filter is hi iff the coefficient is in the right half (cH, cD), the col filter hi iff it is in the bottom half (cV, cD).
        # Coefficients sharing (col filter, q) share v: one A row per distinct (col band, q), B column = sum of c u. Exact.
        seed = int(lparams.get("random_loc_seed", 777))
        scaling = float(lparams.get("scaling", 25.0))
        family = str(lparams.get("wavelet_family", "db1"))
        use_idwt = bool(lparams.get("use_idwt", True))
        if use_idwt:
            lo, hi, red_peft = _wavelet_filters("db1" if family == "haar" else family)
            L = len(lo)
            red = L - 2
            if red_peft is not None and red_peft != red:
                raise ValueError(f"WaveFT {family}: PEFT pads by {red_peft}, filter length {L} implies {red}")
            lo_t, hi_t = torch.tensor(lo, dtype=torch.float64), torch.tensor(hi, dtype=torch.float64)
        for key, p in _module_parts(lora_model, ("waveft_spectrum",)).items():
            spec = p["waveft_spectrum"].to(torch.float64)
            n = int(spec.shape[0])
            M, N = need_shape(key)
            idx = torch.randperm(M * N, generator=torch.Generator().manual_seed(seed))[:n]
            rows, cols = (idx // N).tolist(), (idx % N).tolist()
            groups: dict[tuple[int, int], Tensor] = {}
            vecs: dict[tuple[int, int], Tensor] = {}
            if not use_idwt:
                for k in range(n):
                    g = (0, cols[k])
                    if g not in vecs:
                        vecs[g] = torch.zeros(N, dtype=torch.float64)
                        vecs[g][cols[k]] = 1.0
                        groups[g] = torch.zeros(M, dtype=torch.float64)
                    groups[g][rows[k]] += spec[k] * scaling
            else:
                PM, PN = M + red, N + red
                PM += PM % 2
                PN += PN % 2
                ro, co = (PM - M) // 2, (PN - N) // 2
                H2, W2 = PM // 2, PN // 2
                pad = max(0, (2 * L - 3) // 2)
                full_r, full_c = 2 * (H2 - 1) + L, 2 * (W2 - 1) + L
                sr = (full_r - 2 * pad - M) // 2 + pad  # first kept row of the conv_transpose output
                sc = (full_c - 2 * pad - N) // 2 + pad

                def spread(f: Tensor, pos: int, start: int, size: int) -> Tensor:
                    v = torch.zeros(size, dtype=torch.float64)
                    for i in range(L):
                        t = 2 * pos + i - start
                        if 0 <= t < size:
                            v[t] = f[i]
                    return v

                for k in range(n):
                    pr, pc = rows[k] + ro, cols[k] + co
                    col_hi, row_hi = pr >= H2, pc >= W2
                    pp, qq = pr - H2 * col_hi, pc - W2 * row_hi
                    g = (int(col_hi), qq)
                    if g not in vecs:
                        vecs[g] = spread(hi_t if col_hi else lo_t, qq, sc, N)
                        groups[g] = torch.zeros(M, dtype=torch.float64)
                    groups[g] += (spec[k] * scaling) * spread(hi_t if row_hi else lo_t, pp, sr, M)
            keys_g = sorted(vecs)
            A = torch.stack([vecs[g] for g in keys_g])
            B = torch.stack([groups[g] for g in keys_g], dim=1)
            out[key] = _compact(A, B)
    else:
        raise ValueError(peft_type)
    return {k: (a.to(torch.float32).contiguous(), b.to(torch.float32).contiguous()) for k, (a, b) in out.items()}


def bdlora_expand_state(lora_model: dict[str, Tensor], lparams: dict[str, Any]) -> dict[str, Tensor]:
    """LoRA use_bdlora: on modules matching target_modules_bd_a (resp. _bd_b) PEFT replaces lora_A (resp. lora_B) by a
    BlockDiagonalLinear whose weight [out_f, in_f / nblocks] holds the nblocks diagonal blocks stacked along dim 0
    (lora/variants.py:922-966); the delta is the plain B @ A with that factor as block_diag(*chunk(weight, nblocks, 0)) (:1037-1061).
    The match is a substring test on the module key (:979-986). Densifying the factor gives an exact plain LoRA."""
    bd = lparams["use_bdlora"]
    pa, pb = list(bd.get("target_modules_bd_a") or []), list(bd.get("target_modules_bd_b") or [])
    nb = int(bd.get("nblocks", 1))
    out: dict[str, Tensor] = {}
    n_a = n_b = 0
    for name, t in lora_model.items():
        key = get_peft_module_key(name)
        is_a, is_b = any(s in key for s in pa), any(s in key for s in pb)
        if is_a and is_b:
            raise ValueError(f"use_bdlora: '{key}' matches both the A and the B block-diagonal patterns (PEFT rejects it)")
        if (name.endswith(".lora_A.weight") and is_a) or (name.endswith(".lora_B.weight") and is_b):
            if t.shape[0] % nb:
                raise ValueError(f"use_bdlora '{name}' {tuple(t.shape)}: rows not divisible by nblocks {nb}")
            t = torch.block_diag(*torch.chunk(t, nb, dim=0))
            n_a += name.endswith(".lora_A.weight")
            n_b += name.endswith(".lora_B.weight")
        out[name] = t
    for name, t in out.items():
        if name.endswith(".lora_A.weight"):
            b = out.get(name[: -len(".lora_A.weight")] + ".lora_B.weight")
            if b is not None and b.shape[1] != t.shape[0]:
                raise ValueError(f"use_bdlora '{name}': after densifying, lora_A {tuple(t.shape)} and lora_B {tuple(b.shape)} disagree")
    logger.info(f"use_bdlora: {n_a} lora_A and {n_b} lora_B block-diagonal factor(s) densified (nblocks {nb}, exact)")
    return out


def target_parameter_carriers(lora_model: dict[str, Tensor], lparams: dict[str, Any], dir_base_model: Path | None,
                              hf_model_id: str | None, trust_remote_code: bool) -> tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor, int]]]:
    """Split PEFT target_parameters LoRA (lora.ParamWrapper) out of the state dict (row S69-ADAPTERS-TARGET-PARAMETERS-MOE-EXPERT-LORA
    -2026-10-26-01). Returns (the remaining state dict, HF parameter name -> (A_c, B_c, r)) where the carrier B_c @ A_c has the
    HF PARAMETER's own shape and orientation ([E, s1, s2] for a 3-D expert stack), every PEFT scale folded into B_c, so the base
    model's modify_tensors (transposes, gate/up de-interleave, splits) moves it exactly like the weight it adapts.

    PEFT layout (layer.py:2508-2574,2649-2666): for a parameter of shape [E, s1, s2], lora_A is [r*E, in] with row e*r + k, lora_B is
    [out, r*E] with column k*E + e; in/out are (s2, s1) when the module is NOT marked is_transposed (in/out swapped, :2552-2556),
    else (s1, s2). get_delta_factors gives delta[e] = lhs[e] @ rhs[e] in parameter orientation: swapped lhs = B_e [s1, r],
    rhs = A_e [r, s2]; not swapped lhs = A_e^T, rhs = B_e^T. Several parameters of ONE module nest ParamWrappers: the first
    targeted parameter in the module's registration order is innermost, saved as "<module>.base_layer....lora_A" with the most
    ".base_layer" segments (tuners_utils.py:1218-1311 iterates named_parameters(recurse=False) and wraps the current wrapper)."""
    tps = list(lparams.get("target_parameters") or [])
    if not tps:
        return lora_model, {}
    tp_modules = {t.rpartition(".")[0] for t in tps}

    def is_tp_module(path: str) -> bool:
        return any(path == m or path.endswith("." + m) for m in tp_modules)

    pending: dict[tuple[str, int], dict[str, Tensor]] = {}
    rest: dict[str, Tensor] = {}
    for name, t in lora_model.items():
        key = name.replace("base_model.model.", "", 1)
        m = re.match(r"^(.*?)((?:\.base_layer)*)\.lora_([AB])\.weight$", key)
        if m is None or not is_tp_module(m.group(1)):
            rest[name] = t
            continue
        pending.setdefault((m.group(1), m.group(2).count(".base_layer")), {})[m.group(3)] = t
    if not pending:
        raise ValueError(f"target_parameters {tps} is set but no lora_A/lora_B of those modules is in the adapter")

    # registration order of each module's parameters: from the HF model built on the meta device (no weights, no memory)
    named = dict(_meta_hf_model(dir_base_model, hf_model_id, trust_remote_code).named_modules())

    def find_module(path: str):
        if path in named:
            return path, named[path]
        hits = [k for k in named if k.endswith("." + path) or path.endswith("." + k)]
        if len(hits) != 1:
            raise ValueError(f"target_parameters: module '{path}' matches {len(hits)} modules of the HF model")
        return hits[0], named[hits[0]]

    carriers: dict[str, tuple[Tensor, Tensor, int]] = {}
    for (path, depth), ab in sorted(pending.items()):
        if set(ab) != {"A", "B"}:
            raise ValueError(f"target_parameters '{path}' depth {depth}: have {sorted(ab)}, need lora_A and lora_B")
        mod_name, mod = find_module(path)
        params = [(n, tuple(p.shape)) for n, p in mod.named_parameters(recurse=False)
                  if any(f"{mod_name}.{n}" == t or f"{mod_name}.{n}".endswith("." + t) for t in tps)]
        n_wrapped = len({d for (pp, d) in pending if pp == path})
        if len(params) != n_wrapped:
            raise ValueError(f"target_parameters '{path}': {len(params)} matching parameters {params} but {n_wrapped} nested wrappers saved")
        p_name, shape = params[n_wrapped - 1 - depth]  # innermost (most .base_layer) = first registered
        A, B = ab["A"].to(torch.float64), ab["B"].to(torch.float64)
        hf_key = f"{mod_name}.{p_name}"
        if len(shape) == 2:
            E, s1, s2 = 1, shape[0], shape[1]
            if tuple(A.shape[1:]) != (s2,) or B.shape[0] != s1:
                raise ValueError(f"{hf_key} {shape}: lora_A {tuple(A.shape)} / lora_B {tuple(B.shape)} do not fit a 2-D ParamWrapper")
            r = A.shape[0]
            lhs, rhs = B[None], A[None]
        elif len(shape) == 3:
            E, s1, s2 = shape
            swapped = not bool(getattr(mod, "is_transposed", False))
            n_in, n_out = (s2, s1) if swapped else (s1, s2)
            if A.shape[1] != n_in or B.shape[0] != n_out or A.shape[0] % E or A.shape[0] != B.shape[1]:
                raise ValueError(f"{hf_key} {shape} (in/out {n_in}/{n_out}, experts {E}): lora_A {tuple(A.shape)} / lora_B {tuple(B.shape)} "
                                 f"do not fit PEFT's ParamWrapper layout")
            r = A.shape[0] // E
            A3 = A.reshape(E, r, n_in)                    # [E, r, in]
            B3 = B.reshape(n_out, r, E).permute(2, 0, 1)  # [E, out, r]
            lhs, rhs = (B3, A3) if swapped else (A3.transpose(-1, -2), B3.transpose(-1, -2))
        else:
            raise ValueError(f"{hf_key}: {len(shape)}-D parameter")
        factor = peft_scale_fold_factor(hf_key, int(r), lparams)
        a_c = rhs.contiguous()
        b_c = (lhs * factor).contiguous()
        if E == 1:
            a_c, b_c = a_c[0], b_c[0]
        carriers[hf_key] = (a_c.to(torch.float32), b_c.to(torch.float32), int(r))
        logger.info(f"target_parameters: {hf_key} {shape} <- {'.base_layer' * depth or '(outermost)'} wrapper, rank {r}, "
                    f"{'experts ' + str(E) if E > 1 else '2-D'}, scale fold {factor:g}")
    return rest, carriers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face PEFT LoRA adapter to a GGUF file")
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "auto"], default="f32",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="only print out what will be done, without writing any new files",
    )
    parser.add_argument(
        "--base", type=Path,
        help="directory containing Hugging Face model config files (config.json, tokenizer.json) for the base model that the adapter is based on - only config is needed, actual model weights are not required. If base model is unspecified, it will be loaded from Hugging Face hub based on the adapter config",
    )
    parser.add_argument(
        "--base-model-id", type=str,
        help="the model ID of the base model, if it is not available locally or in the adapter config. If specified, it will ignore --base and load the base model config from the Hugging Face hub (Example: 'meta-llama/Llama-3.2-1B-Instruct')",
    )
    parser.add_argument(
        "--trust-remote-code", default=False, action="store_true",
        help="trust remote code in the model",
    )
    parser.add_argument(
        "--gate", type=str, default=None, choices=["none", "draft_rows", "mask_token"],
        help="position-gated LoRA: write adapter.lora.gate so the runtime adds the delta only on the gated rows (draft_rows: positions at/after "
             "llama_set_draft_start; mask_token: tokens equal to --gate-token). Falls back to adapter_config.json 'gate'. Default: no gate",
    )
    parser.add_argument(
        "--gate-token", type=int, default=None,
        help="token id selecting the gated rows when --gate mask_token (falls back to adapter_config.json 'gate_token')",
    )
    parser.add_argument(
        "lora_path", type=Path,
        help="directory containing Hugging Face PEFT LoRA config (adapter_model.json) and weights (adapter_model.safetensors or adapter_model.bin)",
    )

    return parser.parse_args()


def load_hparams_from_hf(hf_model_id: str, trust_remote_code: bool) -> tuple[dict[str, Any], Path | None]:
    from huggingface_hub import try_to_load_from_cache

    # normally, adapter does not come with base model config, we need to load it from AutoConfig
    config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
    cache_dir = try_to_load_from_cache(hf_model_id, "config.json")
    cache_dir = Path(cache_dir).parent if isinstance(cache_dir, str) else None

    return config.to_dict(), cache_dir


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    ftype = ftype_map[args.outtype]

    dir_base_model: Path | None = args.base
    dir_lora: Path = args.lora_path
    base_model_id: str | None = args.base_model_id
    lora_config = dir_lora / "adapter_config.json"
    input_model = dir_lora / "adapter_model.safetensors"

    if args.outfile is not None:
        fname_out = args.outfile
    else:
        # output in the same directory as the model by default
        fname_out = dir_lora

    if os.path.exists(input_model):
        # lazy import load_file only if lora is in safetensors format.
        from safetensors.torch import load_file

        lora_model = load_file(input_model, device="cpu")
    else:
        input_model = os.path.join(dir_lora, "adapter_model.bin")
        lora_model = torch.load(input_model, map_location="cpu", weights_only=True)

    # load LoRA config
    with open(lora_config, "r") as f:
        lparams: dict[str, Any] = json.load(f)

    # dispatch on peft_type / LoRA variant BEFORE anything else: an adapter with no served kind is refused loudly
    try:
        peft_handler = dispatch_peft_type(lparams)
    except PeftRefusal as e:
        logger.error(str(e))
        sys.exit(EXIT_REFUSED)
    logger.info(f"peft_type dispatch: {str(lparams.get('peft_type') or 'LORA').upper()} -> {peft_handler}")
    if lparams.get("velora_config"):
        logger.info("velora_config: VeLoRA only compresses training activations; the saved A/B are a plain LoRA")

    # load base model
    if base_model_id is not None:
        logger.info(f"Loading base model from Hugging Face: {base_model_id}")
        hparams, dir_base_model = load_hparams_from_hf(base_model_id, args.trust_remote_code)
    elif dir_base_model is None:
        if "base_model_name_or_path" in lparams:
            model_id = lparams["base_model_name_or_path"]
            logger.info(f"Loading base model from Hugging Face: {model_id}")
            try:
                hparams, dir_base_model = load_hparams_from_hf(model_id, args.trust_remote_code)
            except OSError as e:
                logger.error(f"Failed to load base model config: {e}")
                logger.error("Please try downloading the base model and add its path to --base")
                sys.exit(1)
        else:
            logger.error("'base_model_name_or_path' is not found in adapter_config.json")
            logger.error("Base model config is required. Please download the base model and add its path to --base")
            sys.exit(1)
    else:
        logger.info(f"Loading base model: {dir_base_model.name}")
        hparams = ModelBase.load_hparams(dir_base_model, False)

    is_lily = peft_handler == "LILY"
    is_ia3 = peft_handler == "IA3"

    # canonicalise the state dict into plain lora_A / lora_B pairs (+ target_parameters carriers)
    prefolded = False  # True: every PEFT scale is already in B and the GGUF carries lora_alpha 0
    tp_carriers: dict[str, tuple[Tensor, Tensor, int]] = {}
    try:
        if peft_handler in LOWRANK_PEFT_TYPES:
            shapes = base_weight_shapes(dir_base_model) if peft_handler in NEEDS_BASE_SHAPES else {}
            factors = lowrank_tuner_factors(peft_handler, lora_model, lparams, shapes,
                                            TunerBase(dir_base_model, base_model_id, args.trust_remote_code, shapes=shapes or None))
            lora_model = {}
            for key, (a_f, b_f) in factors.items():
                lora_model[f"base_model.model.{key}.lora_A.weight"] = a_f
                lora_model[f"base_model.model.{key}.lora_B.weight"] = b_f
            prefolded = True
            logger.info(f"{peft_handler}: {len(factors)} module(s) rewritten to an exact B @ A (ranks "
                        f"{sorted({int(a.shape[0]) for a, _ in factors.values()})}), scale folded into B, GGUF lora_alpha 0")
        elif peft_handler == "LORA":
            if lparams.get("use_bdlora"):
                lora_model = bdlora_expand_state(lora_model, lparams)
            if lparams.get("use_qalora"):
                lora_model = qalora_expand_state(lora_model, lparams, base_weight_shapes(dir_base_model))
            if lparams.get("target_parameters"):
                if lparams.get("use_dora"):
                    refuse("LoRA target_parameters + use_dora", ROW_TARGET_PARAMETERS, "PEFT's ParamWrapper rejects DoRA (layer.py:2551)")
                lora_model, tp_carriers = target_parameter_carriers(lora_model, lparams, dir_base_model, base_model_id,
                                                                    args.trust_remote_code)
    except PeftRefusal as e:
        logger.error(str(e))
        sys.exit(EXIT_REFUSED)

    with torch.inference_mode():
        model_arch = get_model_architecture(hparams, ModelType.TEXT)
        try:
            model_class = get_model_class(model_arch)
            logger.info("Using model architecture: %s", model_arch)
        except NotImplementedError:
            logger.error(f"Model {model_arch} is not supported")
            sys.exit(1)

        class LoraModel(model_class):  # ty: ignore[unsupported-base]
            model_arch = model_class.model_arch

            lora_alpha: float

            def __init__(self, *args, dir_lora_model: Path, lora_alpha: float, **kwargs):

                super().__init__(*args, **kwargs)

                self.dir_model_card = dir_lora_model
                self.lora_alpha = float(lora_alpha)

            def set_vocab(self):
                pass

            def set_type(self):
                self.gguf_writer.add_type(gguf.GGUFType.ADAPTER)
                # adapter.type stays "lora" (the A/B tensors ARE a LoRA); a DoRA file also lists "dora" in adapter.kinds and
                # carries {base}.dora_m. RESEARCH-DONE: read the tensor-suffix loop src/llama-adapter.cpp:276-298 at fdfaa679e in
                # full: a loader without the DoRA kind throws "LoRA tensor '...' has unexpected suffix" on it (:297; legacy
                # src/llama.cpp:10013 the same), so old binaries refuse a DoRA file loudly.
                self.gguf_writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
                if lparams.get("use_dora", False):
                    self.gguf_writer.add_array(gguf.Keys.Adapter.KINDS, ["lora", "dora"])
                if is_lily:
                    # Lily (row S69-ADAPTERS-W2-LILY-PER-SEQUENCE-ROUTER-2026-10-26-01): tensors {base}.lily_a / .lily_b / .lily_router.
                    # A loader without the kind refuses the file: an unknown kind in adapter.kinds, or (older loaders) the suffix
                    # loop's throw on any tensor name it does not know (src/llama-adapter.cpp:430 at faeb2b3c5).
                    self.gguf_writer.add_array(gguf.Keys.Adapter.KINDS, ["lily"])
                if is_ia3:
                    # IA3 (row S69-ADAPTERS-LINEAR-WRAPPER-IA3-RED-VEC-SCALE-2026-10-26-01): tensors {base}.ia3_in [n_in] /
                    # {base}.ia3_out [n_out], no A/B; src/llama-adapter.cpp loads them under this kind only.
                    self.gguf_writer.add_array(gguf.Keys.Adapter.KINDS, ["ia3"])

            def tensor_force_quant(self, name, new_name, bid, n_dims):
                if new_name.endswith((".ia3_in", ".ia3_out")):
                    return gguf.GGMLQuantizationType.F32  # scale vectors stay F32 at every --outtype (the loader requires F32)
                if new_name.endswith(".lily_router"):
                    return gguf.GGMLQuantizationType.F32  # the router feeds a softmax: kept F32 at every --outtype
                if new_name.endswith(".dora_m"):
                    return gguf.GGMLQuantizationType.F32  # per-row magnitudes stay F32 at every --outtype
                return super().tensor_force_quant(name, new_name, bid, n_dims)

            def set_gguf_parameters(self):
                logger.debug("GGUF KV: %s = %d", gguf.Keys.Adapter.LORA_ALPHA, self.lora_alpha)
                self.gguf_writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, self.lora_alpha)
                gate = args.gate if args.gate is not None else lparams.get("gate")
                if gate is not None and gate != "none":
                    if gate not in ("draft_rows", "mask_token"):
                        raise ValueError(f"unknown adapter gate {gate!r} (none, draft_rows, mask_token)")
                    logger.debug("GGUF KV: %s = %s", gguf.Keys.Adapter.GATE, gate)
                    self.gguf_writer.add_string(gguf.Keys.Adapter.GATE, gate)
                    if gate == "mask_token":
                        gate_token = args.gate_token if args.gate_token is not None else lparams.get("gate_token")
                        if gate_token is None:
                            raise ValueError("--gate mask_token needs --gate-token (or 'gate_token' in adapter_config.json)")
                        self.gguf_writer.add_uint32(gguf.Keys.Adapter.GATE_TOKEN, int(gate_token))
                alora_invocation_tokens = lparams.get("alora_invocation_tokens")
                invocation_string = lparams.get("invocation_string")
                if invocation_string and not alora_invocation_tokens:
                    logger.debug("Tokenizing invocation_string -> alora_invocation_tokens")
                    base_model_path_or_id = hparams.get("_name_or_path")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(base_model_path_or_id)
                    except ValueError:
                        logger.error("Unable to load tokenizer from %s", base_model_path_or_id)
                        raise
                    # NOTE: There's an off-by-one with the older aLoRAs where
                    # the invocation string includes the "<|start_of_turn|>"
                    # token, but the adapters themselves were trained to
                    # activate _after_ that first token, so we drop it here.
                    alora_invocation_tokens = tokenizer(invocation_string)["input_ids"][1:]  # ty: ignore[call-non-callable]
                if alora_invocation_tokens:
                    logger.debug("GGUF KV: %s = %s", gguf.Keys.Adapter.ALORA_INVOCATION_TOKENS, alora_invocation_tokens)
                    self.gguf_writer.add_key_value(
                        gguf.Keys.Adapter.ALORA_INVOCATION_TOKENS,
                        alora_invocation_tokens,
                        GGUFValueType.ARRAY,
                        GGUFValueType.UINT32,
                    )

            def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
                # Never add extra tensors (e.g. rope_freqs) for LoRA adapters
                return ()

            def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
                # PROGRESS-OK: lazy name/shape bookkeeping only; the gguf writer prints per-tensor progress
                tensor_map: dict[str, PartialLoraTensor] = {}
                # base name (after filter_tensors) -> PEFT module key, for rank_pattern/alpha_pattern matching
                tensor_peft_key: dict[str, str] = {}
                self.lora_scale_folded = 0
                # DoRA (row S69-ADAPTERS-DORA-RUNTIME-KIND-2026-10-26-01): base name -> lora_magnitude_vector. NOT routed
                # through PEFT convert_to_lora: measured, it drops the magnitude (findings S69_ENGINE_ADAPTERS_NATIVE_SUPPORT
                # _RESEARCH_2026-10-26.md 2c, probe B diff 7.89).
                dora_m: dict[str, Tensor] = {}
                self.dora_names: set[str] = set()
                use_dora = bool(lparams.get("use_dora", False))
                self.lily_names: set[str] = set()
                self.lily_router: dict[str, Tensor] = {}
                self.lily_num_B = int(lparams.get("num_B", 0))
                lily_parts: dict[str, dict[str, Tensor]] = {}
                self.ia3_sides: dict[str, str] = {}
                if is_ia3:
                    for name, tensor in lora_model.items():
                        if not is_ia3_vector(name):
                            logger.error(f"'{name}' is not an ia3_l tensor of an IA3 adapter")
                            sys.exit(1)
                        base_name = get_base_tensor_name(name)
                        if (titem := self.filter_tensors((base_name, lambda g=tensor: g))) is None:
                            continue
                        side = ia3_side(tuple(tensor.shape), name)
                        if self.lazy:
                            tensor = LazyTorchTensor.from_eager(tensor)
                        self.ia3_sides[titem[0]] = side
                        yield (titem[0], tensor)
                    return
                if is_lily:
                    for name, tensor in lora_model.items():
                        part = lily_part(name)
                        if part is None:
                            logger.error(f"'{name}' is not a lily_A / lily_B / lily_router tensor of a Lily adapter")
                            sys.exit(1)
                        base_name = get_base_tensor_name(name)
                        if (titem := self.filter_tensors((base_name, lambda g=tensor: g))) is None:
                            continue
                        lily_parts.setdefault(titem[0], {})[part] = tensor
                    for base_name, parts in lily_parts.items():
                        if set(parts) != {"A", "B", "router"}:
                            logger.error(f"Lily '{base_name}': have {sorted(parts)}, need A, B and router")
                            sys.exit(1)
                        r_shape = tuple(parts["router"].shape)
                        if r_shape != (self.lily_num_B, parts["A"].shape[0]):
                            logger.error(f"Lily '{base_name}': router {r_shape} != (num_B, r)")
                            sys.exit(1)
                        a_c, b_c = lily_carrier(parts["A"], parts["B"], self.lily_num_B, float(lparams["scaling"]))
                        router = parts["router"].to(torch.float32)
                        if self.lazy:
                            a_c, b_c = LazyTorchTensor.from_eager(a_c), LazyTorchTensor.from_eager(b_c)
                            router = LazyTorchTensor.from_eager(router)
                        self.lily_names.add(base_name)
                        self.lily_router[base_name] = router
                        yield (base_name, cast(torch.Tensor, LoraTorchTensor(a_c, b_c)))
                    return

                for name, tensor in lora_model.items():
                    if self.lazy:
                        tensor = LazyTorchTensor.from_eager(tensor)
                    base_name = get_base_tensor_name(name)
                    peft_key = get_peft_module_key(name)
                    # filter base name, ignore tensor transformations for now
                    data_gen = lambda g=tensor: g  # noqa: E731
                    if (titem := self.filter_tensors((base_name, data_gen))) is None:
                        continue
                    base_name, _ = titem
                    # note: mergekit-extract-lora also adds token embeddings to the adapter
                    is_lora_a = ".lora_A.weight" in name or ".lora_embedding_A" in name
                    is_lora_b = ".lora_B.weight" in name or ".lora_embedding_B" in name
                    if not is_lora_a and not is_lora_b:
                        if ".base_layer.weight" in name:
                            continue
                        if is_dora_magnitude(name):
                            if not use_dora:
                                logger.error(f"'{name}' is a DoRA magnitude but adapter_config.json has use_dora false")
                                sys.exit(1)
                            if ".embed_tokens." in name or "token_embd" in base_name:
                                # PEFT's DoRA embedding variant is a different formula (dora.py DoraEmbeddingLayer: no (d-1)*base term)
                                logger.error(f"'{name}': DoRA on embeddings is not supported by the fork's DoRA kind")
                                sys.exit(1)
                            dora_m[base_name] = tensor
                            continue
                        # mergekit-extract-lora add these layernorm to the adapter, we need to keep them
                        if "_layernorm" in name or ".norm" in name:
                            yield (base_name, tensor)
                            continue
                        logger.error(f"Unexpected name '{name}': Not a lora_A or lora_B tensor")
                        if ".embed_tokens.weight" in name or ".lm_head.weight" in name:
                            logger.error("Embeddings is present in the adapter. This can be due to new tokens added during fine tuning")
                            logger.error("Please refer to https://github.com/ggml-org/llama.cpp/pull/9948")
                        sys.exit(1)

                    tensor_peft_key[base_name] = peft_key  # PROGRESS-OK: dict insert inside the existing loop
                    if base_name in tensor_map:
                        if is_lora_a:
                            tensor_map[base_name].A = tensor
                        else:
                            tensor_map[base_name].B = tensor
                    else:
                        if is_lora_a:
                            tensor_map[base_name] = PartialLoraTensor(A=tensor)
                        else:
                            tensor_map[base_name] = PartialLoraTensor(B=tensor)

                for name, tensor in tensor_map.items():  # PROGRESS-OK: yields lazily to the writer, which reports progress
                    assert tensor.A is not None
                    assert tensor.B is not None
                    lora_b = tensor.B
                    factor = 1.0 if prefolded else peft_scale_fold_factor(tensor_peft_key.get(name, name), int(tensor.A.shape[0]), lparams)
                    if factor != 1.0:
                        lora_b = lora_b.to(torch.float32) * factor
                        self.lora_scale_folded += 1
                    lora_a = tensor.A
                    if name in dora_m:
                        # the fold above already put PEFT's full scaling into B, so the loader's ||W + get_scale * B A|| is
                        # PEFT's ||W + scaling * B A|| (peft 0.17.1 tuners/lora/dora.py:30-35); m itself is never rescaled
                        lora_a, lora_b = dora_carrier(lora_a, lora_b, dora_m.pop(name), self.lazy)
                        self.dora_names.add(name)
                    elif use_dora:
                        logger.error(f"use_dora is true but '{name}' has no lora_magnitude_vector")
                        sys.exit(1)
                    yield (name, cast(torch.Tensor, LoraTorchTensor(lora_a, lora_b)))
                if dora_m:
                    logger.error(f"DoRA magnitude(s) without a lora_A/lora_B pair: {sorted(dora_m)}")
                    sys.exit(1)
                # target_parameters carriers: B_c @ A_c has the HF parameter's own shape; the PEFT scale is already in B_c
                for hf_key, (a_c, b_c, _r) in tp_carriers.items():
                    if (titem := self.filter_tensors((hf_key, lambda g=a_c: g))) is None:
                        logger.error(f"target_parameters '{hf_key}' was dropped by the base model's filter_tensors")
                        sys.exit(1)
                    if self.lazy:
                        a_c, b_c = LazyTorchTensor.from_eager(a_c), LazyTorchTensor.from_eager(b_c)
                    yield (titem[0], cast(torch.Tensor, LoraTorchTensor(a_c, b_c)))

            def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
                if name in self.ia3_sides:
                    side = self.ia3_sides[name]
                    # an (out, 1) vector rides the base class's own row permutations (q/k rotary reorder, fused qkv splits) exactly
                    # as the weight's rows do; an (1, in) vector has no row axis to permute, so a q/k INPUT scale (a feedforward
                    # module that is a q/k projection) stops with an error rather than being mis-permuted
                    try:
                        ia3_dest = list(super().modify_tensors(data_torch, name, bid))
                    except (RuntimeError, ValueError, AssertionError) as e:
                        logger.error(f"IA3 '{name}' ({side} scale): the base model's tensor rewrite cannot take a vector: {e}")
                        sys.exit(1)
                    for dest_name, dest_data in ia3_dest:
                        if not dest_name.endswith(".weight"):
                            logger.error(f"IA3 '{name}' maps to '{dest_name}', which is not a weight")
                            sys.exit(1)
                        yield (dest_name + (".ia3_in" if side == "in" else ".ia3_out"), dest_data.reshape(-1))
                    return
                dest = list(super().modify_tensors(data_torch, name, bid))
                # some archs may have the same tensor for lm_head and output (tie word embeddings)
                # in this case, adapters targeting lm_head will fail when using llama-export-lora
                # therefore, we ignore them for now
                # see: https://github.com/ggml-org/llama.cpp/issues/9065
                if name == "lm_head.weight" and len(dest) == 0:
                    raise ValueError("lm_head is present in adapter, but is ignored in base model")
                for dest_name, dest_data in dest:
                    # mergekit-extract-lora add these layernorm to the adapter
                    if "_norm" in dest_name:
                        assert dest_data.dim() == 1
                        yield (dest_name, dest_data)
                        continue

                    # otherwise, we must get the lora_A and lora_B tensors
                    assert isinstance(dest_data, LoraTorchTensor)
                    lora_a, lora_b = dest_data.get_lora_A_B()

                    # note: mergekit-extract-lora flip and transpose A and B
                    # here we only need to transpose token_embd.lora_a, see llm_build_inp_embd()
                    if "token_embd.weight" in dest_name:
                        lora_a = lora_a.T

                    if name in self.lily_names:
                        # undo lily_carrier(): A = the first r rows of A_c; B_c [n_out, num_B * r] -> [num_B, n_out, r] (GGUF ne
                        # [r, n_out, num_B], what llama_lily_delta reshapes to [r, n_out * num_B])
                        n_b = self.lily_num_B
                        r = lora_a.shape[0] // n_b
                        n_out = lora_b.shape[0]
                        yield (dest_name + ".lily_a", lora_a[:r, :])
                        yield (dest_name + ".lily_b", lora_b.reshape(n_out, n_b, r).permute(1, 0, 2).contiguous())
                        yield (dest_name + ".lily_router", self.lily_router[name])
                        continue

                    if name in self.dora_names:
                        # strip the magnitude column dora_carrier() appended (it rode through every row op with B)
                        dora_m_out = lora_b[..., -1]
                        lora_b = lora_b[..., :-1]
                        lora_a = lora_a[..., :-1, :]
                        yield (dest_name + ".dora_m", dora_m_out)

                    yield (dest_name + ".lora_a", lora_a)
                    yield (dest_name + ".lora_b", lora_b)

        # a Lily adapter has no lora_alpha: its scaling is folded into lily_b, and alpha 0 makes the loader's get_scale = the user scale
        # the same for every tuner rewritten by lowrank_tuner_factors (prefolded)
        alpha: float = 0.0 if (is_lily or is_ia3 or prefolded) else lparams["lora_alpha"]

        model_instance = LoraModel(
            dir_base_model,
            ftype,
            fname_out,
            is_big_endian=args.bigendian,
            use_temp_file=False,
            eager=args.no_lazy,
            dry_run=args.dry_run,
            dir_lora_model=dir_lora,
            lora_alpha=alpha,
            hparams=hparams,
            remote_hf_model_id=base_model_id,
        )

        logger.info("Exporting model...")
        model_instance.write()
        n_folded = getattr(model_instance, "lora_scale_folded", 0)
        if n_folded:
            logger.info(f"PEFT scale (use_rslora / alpha_pattern) folded into lora_b for {n_folded} tensor(s)")
        logger.info(f"Model successfully exported to {model_instance.fname_out}")
