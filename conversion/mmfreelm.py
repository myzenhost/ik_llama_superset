from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("HGRNBitForCausalLM")
@ModelBase.example("ridger/MMfreeLM-370M")
class MMfreeLMModel(TextModel):
    """MatMul-free LM (arXiv 2406.02528): MLGRU token mixer + GLU channel mixer, every projection a BitLinear.

    The checkpoint stores latent full-precision weights; BitLinear ternarizes them on the fly with ONE absmean scale per
    weight tensor. That ternarization is done here (same formula as mmfreelm/ops/fusedbitnet.py weight_quant), so the
    GGUF holds exact ternary matrices as TQ2_0 (per-256 block scale == the tensor scale). gate_proj is ternarized as one
    [2*ff, hidden] tensor (one shared scale) and then split into ffn_gate / ffn_up. lm_head is a BitLinear as well.
    The cumulative-softmax lower bound of the forget gate depends only on the stored `lower_bounds` parameter, so it is
    precomputed per layer.
    """
    model_arch = gguf.MODEL_ARCH.MMFREELM

    _TERNARY_SUFFIXES = (
        "hgrn_i.weight", "hgrn_f.weight", "hgrn_g.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    )

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        hidden = self.hparams["hidden_size"]
        ff = self.hparams.get("intermediate_size")
        if ff is None:
            # HGRNBitMLP: multiple of 256 closest above 2/3 * hidden * hidden_ratio
            ff = int(hidden * self.hparams.get("hidden_ratio", 4) * 2 / 3)
            ff = 256 * ((ff + 256 - 1) // 256)
        if self.hparams.get("use_short_conv", False):
            raise ValueError("mmfreelm: use_short_conv=True is not supported (released checkpoints use False)")
        if self.hparams.get("num_heads", 1) != 1 or self.hparams.get("expand_ratio", 1) != 1:
            raise ValueError("mmfreelm: only num_heads=1, expand_ratio=1 are supported")
        if not self.hparams.get("use_lower_bound", True):
            raise ValueError("mmfreelm: use_lower_bound=False is not supported")
        self.gguf_writer.add_context_length(self.hparams.get("max_position_embeddings", 2048))  # recurrent: not a hard limit
        self.gguf_writer.add_embedding_length(hidden)
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_feed_forward_length(ff)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])
        self.gguf_writer.add_head_count(0)  # required by the loader, unused
        self.gguf_writer.add_file_type(self.ftype)
        self._ff = ff

    @staticmethod
    def _ternarize(w: Tensor) -> Tensor:
        w = w.float()
        s = w.abs().mean().clamp(min=1e-5)
        return (w / s).round().clamp(-1, 1) * s

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if n_dims == 2 and new_name.endswith(self._TERNARY_SUFFIXES + ("output.weight",)) and not new_name.startswith("token_embd"):
            return gguf.GGMLQuantizationType.TQ2_0
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.embeddings.weight":
            yield "token_embd.weight", data_torch
            return
        if name == "model.norm.weight":
            yield "output_norm.weight", data_torch.float()
            return
        if name == "lm_head.norm.weight":
            yield "output_in_norm.weight", data_torch.float()
            return
        if name == "lm_head.weight":
            yield "output.weight", self._ternarize(data_torch)
            return
        if name == "model.lower_bounds":
            lb = data_torch.float().softmax(0)
            lb = lb.cumsum(0) - lb[0]
            for i in range(lb.shape[0]):
                yield f"blk.{i}.hgrn_lower_bound.weight", lb[i].contiguous()
            return

        assert bid is not None, name
        tail = name.split(f"model.layers.{bid}.", 1)[1]
        b = f"blk.{bid}."
        simple = {
            "attn_norm.weight": "attn_norm.weight",
            "mlp_norm.weight": "ffn_norm.weight",
            "attn.g_norm.weight": "hgrn_gnorm.weight",
            "attn.i_proj.norm.weight": "hgrn_i_norm.weight",
            "attn.f_proj.norm.weight": "hgrn_f_norm.weight",
            "attn.g_proj.norm.weight": "hgrn_g_norm.weight",
            "attn.o_proj.norm.weight": "hgrn_o_norm.weight",
            "mlp.gate_proj.norm.weight": "ffn_in_norm.weight",
            "mlp.down_proj.norm.weight": "ffn_down_in_norm.weight",
        }
        ternary = {
            "attn.i_proj.weight": "hgrn_i.weight",
            "attn.f_proj.weight": "hgrn_f.weight",
            "attn.g_proj.weight": "hgrn_g.weight",
            "attn.o_proj.weight": "attn_output.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        if tail in simple:
            yield b + simple[tail], data_torch.float()
        elif tail in ternary:
            yield b + ternary[tail], self._ternarize(data_torch)
        elif tail == "mlp.gate_proj.weight":
            q = self._ternarize(data_torch)  # one shared scale over both halves, as in the reference
            ff = q.shape[0] // 2
            # HGRNBitMLP: gate, y = gate_proj(x).chunk(2, -1); out = down(silu(gate) * y)
            yield b + "ffn_gate.weight", q[:ff].contiguous()
            yield b + "ffn_up.weight", q[ff:].contiguous()
        else:
            raise ValueError(f"mmfreelm: unmapped tensor {name!r}")
