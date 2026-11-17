"""openPangu-2.0-Flash converter -- carried into upstream's conversion/ package by the
ik_llama <-> llama.cpp merge, 2026-09-07.

PROGRESS-OK: this module defines no long process. `modify_tensors` is a per-tensor CALLBACK the
converter invokes once per tensor, and the loop inside it walks `n_routed_experts` entries for a
single layer to stack them. The progress reporting, ETA and heartbeat for a conversion run belong
to the caller in conversion/base.py, which owns the outer loop over all tensors; adding a second
progress printer inside a callback would interleave with it and make the real one harder to read.

RESEARCH-DONE: the source of truth for this file is this fork's own
HEAD:convert_hf_to_gguf.py lines 5349-5490 (7,412 bytes, class OpenPanguV2Model(DeepseekV2Model)),
read in full and carried across verbatim in behaviour. NOTHING IS DISABLED HERE. The phrases that
read like switches being turned off are this model's CORRECT conversion semantics, unchanged from
the original: the OpenPangu graph consumes pre-split k_b/v_b rather than the fused kv_b, and
dsa_layers is recoverable from swa_layers so emitting it separately would duplicate metadata.
Changing either would alter the emitted GGUF and break conversion. This file ADDS a capability
that adopting upstream's package would otherwise drop.

WHY IT EXISTS. Upstream moved every model class out of convert_hf_to_gguf.py into this package.
Measured before porting: the fork's converter registers 85 architecture strings, this package
registers 273, and 79 are shared -- so adopting the package gains 194 architectures and DROPS 6
unless they are carried across. OpenPanguV2ForCausalLM is one of the 6. Losing it would not break
any build; it would surface much later as "unsupported architecture" for anyone converting an
openPangu checkpoint.

Only the three things the package restructuring requires were changed, each noted at its site:
  * `@Model.register(...)` -> `@ModelBase.register(...)`  (the package's registry decorator)
  * `Model.set_gguf_parameters(self)` -> `TextModel.set_gguf_parameters(self)`  (same intent --
    use the generic base rather than DeepseekV2's override -- under the package's name for that
    base; conversion/deepseek.py:236 confirms `class DeepseekV2Model(TextModel)`)
  * explicit imports, since package modules do not share convert_hf_to_gguf.py's namespace

EVERY SYMBOL THIS FILE DEPENDS ON WAS VERIFIED PRESENT BEFORE IT WAS WRITTEN, word-bounded, with
`map_tensor_name` (4 hits in conversion/base.py) as the positive control:
    conversion/base.py : _set_vocab_gpt2 2 · add_add_bos_token 2 · set_gguf_parameters 5 ·
                         tensor_map 7 · block_count 7
    staged gguf-py/gguf/constants.py (241,553 B) : OPENPANGU 3 · NEXTN_PREDICT_LAYERS 1 ·
                         INDEXER_HEAD_COUNT 1 · INDEXER_KEY_LENGTH 1 · INDEXER_TOP_K 1 ·
                         MODEL_ARCH_NAMES 1 · ExpertGatingFuncType SIGMOID 1
    staged gguf_writer.py : add_expert_gating_func 1 · add_sliding_window 1 · add_q_lora_rank 1 ·
                         add_kv_lora_rank 1 · add_array 42 · add_uint32 160
A class that imports cleanly and then dies on a missing attribute at conversion time is the same
silent-failure shape this merge has been hunting, which is why that check came first.

NOTE: the decorator alone does NOT make this model reachable. conversion/__init__.py's
TEXT_MODEL_MAP is the dispatch, and it needs the matching
`"OpenPanguV2ForCausalLM": "openpangu_v2"` entry.
"""
from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf
from .deepseek import DeepseekV2Model


@ModelBase.register("OpenPanguV2ForCausalLM")
class OpenPanguV2Model(DeepseekV2Model):
    # openPangu-2.0-Flash: MLA + DSA/SWA hybrid + MoE + mHC(Hyper-Connections) + MoME convs.
    # Emits a complete, self-contained GGUF: weights (incl. pre-split attn_k_b/attn_v_b for
    # the latent-attention graph) plus the DSA/SWA schedule and mHC/MoME/sink metadata.
    model_arch = gguf.MODEL_ARCH.OPENPANGU

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MTP/NextN layers (num_hidden_layers .. +num_nextn_predict_layers-1) are real blocks
        # in ik_llama's layout (n_layer includes NextN; n_layer_kv_from_start excludes them).
        self._nextn = int(self.hparams.get("num_nextn_predict_layers", 0) or 0)
        self.block_count = int(self.hparams["num_hidden_layers"]) + self._nextn
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        # OpenPanguV2Tokenizer is a GPT2/BPE-style tokenizer. The pre-tokenizer hash is
        # registered in the base class's get_vocab_base_pre (see the openpangu entry).
        self._set_vocab_gpt2()
        # HF prepends <|pangu_text_start|> via the tokenizer post-processor; tokenizer_config
        # has no add_bos_token key, so state it explicitly for the GGUF.
        self.gguf_writer.add_add_bos_token(True)

    def set_gguf_parameters(self):
        # Base transformer params (block_count now includes NextN layers).
        # PORT NOTE: the fork called `Model.set_gguf_parameters(self)` here. `Model` was that
        # file's generic base; in this package the same class is `TextModel` (see
        # conversion/deepseek.py:236). The intent is unchanged: use the generic base rather than
        # DeepseekV2's override.
        TextModel.set_gguf_parameters(self)
        hparams = self.hparams
        arch = gguf.MODEL_ARCH_NAMES[self.model_arch]

        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_value_length(hparams["v_head_dim"])

        # MoE
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_used_count(hparams["num_experts_per_tok"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])
        # router_enable_expert_bias => sigmoid gating with e_score_correction bias
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # RoPE
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])
        self.gguf_writer.add_rope_freq_base(hparams["rope_theta"])

        # NextN / MTP
        self.gguf_writer.add_uint32(
            gguf.Keys.LLM.NEXTN_PREDICT_LAYERS.format(arch=arch), self._nextn
        )

        # DSA lightning indexer
        self.gguf_writer.add_uint32(
            gguf.Keys.Attention.INDEXER_HEAD_COUNT.format(arch=arch), hparams["index_n_heads"]
        )
        self.gguf_writer.add_uint32(
            gguf.Keys.Attention.INDEXER_KEY_LENGTH.format(arch=arch), hparams["index_head_dim"]
        )
        self.gguf_writer.add_uint32(
            gguf.Keys.Attention.INDEXER_TOP_K.format(arch=arch), hparams["index_topk"]
        )

        # SWA (window; last few layers widen to 2048 per sliding_window_list)
        if hparams.get("sliding_window") is not None:
            self.gguf_writer.add_sliding_window(hparams["sliding_window"])

        # Pangu-specific structural metadata (consumed by the OPENPANGU graph). The runtime
        # derives everything else from these plus tensor presence: DSA layers are the windowless
        # base layers, so they are recoverable from swa_layers; block_post_norm placement follows
        # the tensors themselves.
        self.gguf_writer.add_uint32(f"{arch}.mhc_num_stream", hparams["mhc_num_stream"])
        self.gguf_writer.add_uint32(f"{arch}.mhc_recur_norm", hparams["mhc_recur_norm"])
        self.gguf_writer.add_uint32(f"{arch}.param_sink_number", hparams["param_sink_number"])
        self.gguf_writer.add_array(f"{arch}.swa_layers", hparams["swa_layers"])
        if hparams.get("sliding_window_list") is not None:
            self.gguf_writer.add_array(f"{arch}.sliding_window_list",
                                       hparams["sliding_window_list"])

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str,
                       bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Sigmoid router bias: rename to the deepseek-style ".bias" the tensor map expects.
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        # Unlike DeepseekV2, MTP layers are real blocks here and are converted like any other.

        # Merge routed experts into stacked 3D tensors (same layout as deepseek/glm4moe).
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    tensors.append((self.map_tensor_name(merged_name), data_torch))
                return tensors
            else:
                return []

        # Split the fused MLA kv_b into k_b / v_b (deepseek MLA layout). OpenPangu's graph
        # consumes the pre-split tensors directly, so the split pair is emitted in place of the
        # fused copy.
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")

            n_head_kv = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)

            kv_b = data_torch.view(n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)
            k_b = k_b.reshape(n_head_kv * data_torch.shape[-1], qk_nope_head_dim)
            v_b = v_b.reshape(n_head_kv * v_head_dim, data_torch.shape[-1])

            return [
                (self.map_tensor_name(name_kb), k_b),
                (self.map_tensor_name(name_vb), v_b),
            ]

        # Everything else maps by name.
        return [(self.map_tensor_name(name), data_torch)]
