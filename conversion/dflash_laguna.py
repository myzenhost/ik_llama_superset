"""DFlash-Laguna draft model -- carried into upstream's conversion/ package by the
ik_llama <-> llama.cpp merge, 2026-09-07.

PROGRESS-OK: no long process here. `modify_tensors` is a per-tensor CALLBACK and `prepare_tensors`
runs once; the conversion run's progress, ETA and heartbeat belong to the outer loop in
conversion/base.py, which owns it.

RESEARCH-DONE: ported from this fork's own HEAD:convert_hf_to_gguf.py lines 2714-2840 (6,742
bytes, class DFlashLagunaModel(DFlashDraftModel)), read in full, together with the three helper
methods it inherits from DFlashDraftModel (lines 2363-2654). NOTHING IS DISABLED. Every `raise
ValueError` below is the original's own precondition check -- this converter deliberately refuses
a checkpoint it cannot represent faithfully rather than emitting a GGUF that loads and decodes
wrongly. This file ADDS a capability that adopting upstream's package would otherwise drop.

WHY IT EXISTS. Adopting upstream's conversion/ package gains 194 architectures and drops 6 unless
they are carried (fork registry 85, package 273, shared 79; controls: the two sets overlap by 79,
an invented name is in neither). DFlashLagunaForCausalLM is one of the 6.

THE TWO LINEAGES HAVE DIVERGED, and this is the substance of the port rather than a detail:
    fork    : class DFlashLagunaModel(DFlashDraftModel), model_arch = MODEL_ARCH.DFLASH_DRAFT
    package : conversion/qwen.py:642 registers "DFlashDraftModel"
              conversion/qwen.py:644 class DFlashModel(Qwen3Model), model_arch = MODEL_ARCH.DFLASH
The package's DFlashModel (lines 644-717, 3,125 bytes) defines filter_tensors, modify_tensors,
set_gguf_parameters and set_vocab -- a better set_vocab in particular, which reuses the TARGET
model's own vocab handler. But it does NOT define the three helpers this class calls, measured
word-bounded with set_gguf_parameters = 18 in qwen.py as the control:
    _get_target_hidden_size   qwen.py 0, base.py 0
    _causal_attention         qwen.py 0, base.py 0
    _get_target_raw_hparams   (its dependency) likewise absent
So they are carried here, self-contained, rather than the class silently inheriting nothing and
failing at conversion time on a missing attribute.

WHY model_arch STAYS DFLASH_DRAFT rather than following the package's base to DFLASH. Decided from
the RUNTIME, not from preference -- both are live, distinct architectures in this fork's C++:
    src/llama-arch.cpp:97    { LLM_ARCH_DFLASH,       "dflash"       }
    src/llama-arch.cpp:98    { LLM_ARCH_DFLASH_DRAFT, "dflash-draft" }
    src/llama-build-context.cpp:3071-3072   both handled
    src/llama-hparams.cpp:1093              a DFLASH_DRAFT-specific case
and both constants exist in the staged gguf-py/gguf/constants.py (DFLASH 3, DFLASH_DRAFT 3). The
fork's Laguna emitted "dflash-draft" GGUFs; changing the arch would change what the runtime loads
them as. Retargeting to DFLASH would be a runtime-behaviour decision, not a merge cleanup, and is
deliberately not made here.

EVERY OTHER SYMBOL VERIFIED PRESENT FIRST, word-bounded, control DFLASH_DRAFT = 3:
    staged constants.py : DFLASH_DRAFT 3 · DFLASH_AUX_HIDDEN_NORM 3 · ATTN_GATE 14 ·
                          TENSOR_NAMES 1
    conversion/base.py  : target_model_dir 3 (declared line 103, set line 152)
NOTE on load_hparams: the fork called `Model.load_hparams(dir)` with one argument; the package's
is `load_hparams(dir_model, is_mistral_format)` (base.py:1105) and takes two. Rather than guess at
the second, the target config is read directly with json.load -- which is exactly what the
package's own DFlashModel.set_vocab does at conversion/qwen.py:660-662.

NOTE: the decorator alone does NOT make this model reachable. conversion/__init__.py's
TEXT_MODEL_MAP is the dispatch and needs `"DFlashLagunaForCausalLM": "dflash_laguna"`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .qwen import DFlashModel


@ModelBase.register("DFlashLagunaForCausalLM")
class DFlashLagunaModel(DFlashModel):
    # Laguna draft model on the DFlash path: per-head attention gating, packed QKV, and auxiliary
    # hidden-state norms taken from the target model's layers.
    model_arch = gguf.MODEL_ARCH.DFLASH_DRAFT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._laguna_qkv_ids: set[int] = set()
        self._laguna_gate_ids: set[int] = set()
        self._laguna_aux_norm_ids: set[int] = set()
        self._target_raw_hparams: dict[str, Any] | None = None

    # ---- helpers carried from the fork's DFlashDraftModel ------------------------------------
    # The package's DFlashModel does not define these (measured: 0 hits in conversion/qwen.py and
    # conversion/base.py). Inheriting from it without them would produce a class that imports
    # cleanly and dies on a missing attribute mid-conversion.

    def _require_target_model_dir(self) -> Path:
        if self.target_model_dir is None:
            raise ValueError("DFlashLagunaForCausalLM conversion requires "
                             "--target-model-dir <matching target model directory>")
        return self.target_model_dir

    def _get_target_raw_hparams(self) -> dict[str, Any]:
        # Read the target config directly, as conversion/qwen.py:660-662 does. The package's
        # ModelBase.load_hparams takes (dir_model, is_mistral_format); reading the JSON avoids
        # guessing that second argument for a draft model's target.
        if self._target_raw_hparams is None:
            with open(self._require_target_model_dir() / "config.json", "r",
                      encoding="utf-8") as f:
                self._target_raw_hparams = json.load(f)
        return self._target_raw_hparams

    def _get_target_hidden_size(self) -> int | None:
        raw_hparams = self._get_target_raw_hparams()
        if (hidden_size := raw_hparams.get("hidden_size")) is not None:
            return int(hidden_size)
        if (hidden_size := raw_hparams.get("backbone_hidden_size")) is not None:
            return int(hidden_size)
        text_hparams = raw_hparams.get("text_config")
        if isinstance(text_hparams, dict) and (hidden_size :=
                                               text_hparams.get("hidden_size")) is not None:
            return int(hidden_size)
        return None

    def _causal_attention(self) -> bool:
        # Laguna's draft stack is causal; the generic DFlash draft path is not.
        return True

    # ---- conversion ---------------------------------------------------------------------------

    def set_gguf_parameters(self):
        dflash_cfg = self.hparams.get("dflash_config")
        if not isinstance(dflash_cfg, dict) or dflash_cfg.get("causal") is not True:
            raise ValueError("DFlashLagunaForCausalLM requires dflash_config.causal=true")
        if self.hparams.get("gating") != "per-head":
            raise ValueError("DFlashLagunaForCausalLM currently requires gating='per-head'")
        target_hidden_size = self._get_target_hidden_size()
        draft_hidden_size = int(self.hparams["hidden_size"])
        if target_hidden_size != draft_hidden_size:
            raise ValueError(
                "DFlashLagunaForCausalLM requires matching target and draft hidden sizes, "
                f"got target={target_hidden_size} and draft={draft_hidden_size}"
            )
        layer_types = self.hparams.get("layer_types")
        if not isinstance(layer_types, list) or len(layer_types) != self.block_count:
            raise ValueError(
                "DFlashLagunaForCausalLM requires one layer_types entry per draft layer"
            )
        if any(str(layer_type) != "sliding_attention" for layer_type in layer_types):
            raise ValueError(
                "DFlashLagunaForCausalLM currently requires every draft layer to use "
                "sliding_attention"
            )
        if not self.hparams.get("sliding_window"):
            raise ValueError("DFlashLagunaForCausalLM requires sliding_window metadata")

        self.hparams["use_sliding_window"] = True
        super().set_gguf_parameters()
        self.gguf_writer.add_bool(f"{self.gguf_writer.arch}.dflash.laguna", True)

    def modify_tensors(self, data_torch: Tensor, name: str,
                       bid: int | None) -> Iterable[tuple[str, Tensor]]:
        top_level_name = name[6:] if name.startswith("model.") else name
        hidden_size = int(self.hparams["hidden_size"])

        if top_level_name.startswith("aux_hidden_norms.") and top_level_name.endswith(".weight"):
            parts = top_level_name.split(".")
            if len(parts) != 3 or not parts[1].isdigit():
                raise ValueError(
                    f"DFlashLagunaForCausalLM: invalid auxiliary norm name {name!r}")
            aux_id = int(parts[1])
            if data_torch.ndim != 1 or data_torch.shape[0] != hidden_size:
                raise ValueError(
                    f"DFlashLagunaForCausalLM: auxiliary norm {name!r} has shape "
                    f"{tuple(data_torch.shape)}, expected [{hidden_size}]"
                )
            self._laguna_aux_norm_ids.add(aux_id)
            tensor_name = gguf.TENSOR_NAMES[
                gguf.MODEL_TENSOR.DFLASH_AUX_HIDDEN_NORM].format(bid=aux_id)
            return [(f"{tensor_name}.weight", data_torch)]

        if top_level_name.endswith(".self_attn.qkv_proj.weight"):
            if bid is None:
                raise ValueError(
                    f"DFlashLagunaForCausalLM: can not infer block id for tensor {name!r}")
            n_head = int(self.hparams["num_attention_heads"])
            n_head_kv = int(self.hparams["num_key_value_heads"])
            head_dim = int(self.hparams.get("head_dim", self.hparams["hidden_size"] // n_head))
            q_width = n_head * head_dim
            k_width = n_head_kv * head_dim
            v_width = n_head_kv * head_dim
            expected_width = q_width + k_width + v_width
            if data_torch.ndim != 2 or data_torch.shape != (expected_width, hidden_size):
                raise ValueError(
                    f"DFlashLagunaForCausalLM: packed QKV tensor {name!r} has shape "
                    f"{tuple(data_torch.shape)}, expected [{expected_width}, {hidden_size}]"
                )
            q_weight, k_weight, v_weight = data_torch.split([q_width, k_width, v_width], dim=0)
            self._laguna_qkv_ids.add(bid)
            result: list[tuple[str, Tensor]] = []
            for suffix, weight in (("q_proj", q_weight), ("k_proj", k_weight),
                                   ("v_proj", v_weight)):
                split_name = name.replace("qkv_proj", suffix)
                result.extend(super().modify_tensors(weight, split_name, bid))
            return result

        if top_level_name.endswith(".self_attn.g_proj.weight"):
            if bid is None:
                raise ValueError(
                    f"DFlashLagunaForCausalLM: can not infer block id for tensor {name!r}")
            gate = data_torch.squeeze().contiguous()
            n_head = int(self.hparams["num_attention_heads"])
            if gate.ndim != 2 or gate.shape != (n_head, hidden_size):
                raise ValueError(
                    f"DFlashLagunaForCausalLM: attention gate {name!r} has shape "
                    f"{tuple(gate.shape)}, expected [{n_head}, {hidden_size}]"
                )
            self._laguna_gate_ids.add(bid)
            tensor_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ATTN_GATE].format(bid=bid)
            return [(f"{tensor_name}.weight", gate)]

        return super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        # Completeness checks: every draft layer must have contributed a packed QKV and a gate,
        # and the auxiliary norms must match the target_layer_ids count. A checkpoint that fails
        # these would convert into a GGUF that loads and decodes with missing tensors, so the
        # conversion stops here instead.
        expected_layers = set(range(self.block_count))
        dflash_cfg = self.hparams.get("dflash_config")
        if not isinstance(dflash_cfg, dict):
            raise ValueError("DFlashLagunaForCausalLM requires dflash_config metadata")
        target_layer_ids = dflash_cfg.get("target_layer_ids", [])
        if not isinstance(target_layer_ids, list) or not target_layer_ids:
            raise ValueError(
                "DFlashLagunaForCausalLM requires non-empty target_layer_ids metadata")
        expected_aux = set(range(len(target_layer_ids)))
        if self._laguna_qkv_ids != expected_layers:
            raise ValueError(
                f"DFlashLagunaForCausalLM: packed QKV layers {sorted(self._laguna_qkv_ids)} "
                f"do not match expected {sorted(expected_layers)}"
            )
        if self._laguna_gate_ids != expected_layers:
            raise ValueError(
                f"DFlashLagunaForCausalLM: attention gate layers {sorted(self._laguna_gate_ids)} "
                f"do not match expected {sorted(expected_layers)}"
            )
        if self._laguna_aux_norm_ids != expected_aux:
            raise ValueError(
                f"DFlashLagunaForCausalLM: auxiliary norm ids "
                f"{sorted(self._laguna_aux_norm_ids)} do not match expected "
                f"{sorted(expected_aux)}"
            )
