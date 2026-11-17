from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("BitnetForCausalLM", "BitNetForCausalLM")
@ModelBase.example("microsoft/bitnet-b1.58-2B-4T")
class BitnetModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BITNET

    def set_vocab(self):
        # The 1bitLLM/bitnet_b1_58-* releases ship a Llama-2 SentencePiece tokenizer.model; microsoft/bitnet-b1.58-2B-4T
        # ships only a Llama-3 byte-level BPE tokenizer.json, and the SentencePiece path cannot convert it at all
        # (FileNotFoundError on tokenizer.model). The BPE path hash-detects the pre-tokenizer (llama-bpe for 2B-4T), so the
        # GGUF carries tokenizer.ggml.pre instead of falling back to 'default' at load.
        if (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        else:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # 1bitLLM checkpoints use SiLU, microsoft/bitnet-b1.58-2B-4T uses relu2 (squared ReLU); the runtime reads this key
        # and treats a file without it as SiLU.
        if (hidden_act := self.hparams.get("hidden_act")) is not None:
            if hidden_act not in ("silu", "relu2"):
                raise ValueError(f"bitnet: unsupported hidden_act {hidden_act!r} (expected silu or relu2)")
            self.gguf_writer.add_hidden_act(hidden_act)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)

    def weight_quant(self, weight: Tensor) -> Tensor:
        dtype = weight.dtype
        weight = weight.float()
        scale = weight.abs().mean().clamp(min=1e-5)
        iscale = 1 / scale
        # TODO: multiply by the scale directly instead of inverting it twice
        # (this is also unnecessarily doubly inverted upstream)
        # ref: https://huggingface.co/1bitLLM/bitnet_b1_58-3B/blob/af89e318d78a70802061246bf037199d2fb97020/utils_quant.py#L10
        result = (weight * iscale).round().clamp(-1, 1) / iscale
        return result.type(dtype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        new_name = self.map_tensor_name(name)

        # A pre-quantized checkpoint (quant_method "bitnet", e.g. 2B-4T) arrives here already dequantized by
        # dequant_bitnet to exactly t*k (t ternary, k the per-tensor weight_scale or its inverse), which is what
        # transformers computes with. weight_quant is not idempotent on such a tensor: it rescales by mean|t*k|, so the
        # result would be t*k*mean|t|, shrinking q, k, o and down by mean|t| (~0.5-0.7) and flattening attention.
        quant_config = self.hparams.get("quantization_config")
        prequantized = isinstance(quant_config, dict) and quant_config.get("quant_method") == "bitnet"

        # A LoRA adapter (convert_lora_to_gguf.py hands this class a LoraTorchTensor A/B pair) is a full-precision delta added on
        # top of the ternary base product; ternarizing it is wrong, and LoraTorchTensor has no .float() so it crashed
        # (MEASURED 2026-10-26, row S69-ADAPTERS-RAW-MATMUL-LORA-BYPASS-MLA-BITNET-2026-10-26-01).
        is_lora_pair = hasattr(data_torch, "get_lora_A_B")

        if not prequantized and not is_lora_pair and any(self.match_model_tensor_name(new_name, key, bid) for key in [
            gguf.MODEL_TENSOR.ATTN_Q,
            gguf.MODEL_TENSOR.ATTN_K,
            gguf.MODEL_TENSOR.ATTN_V,
            gguf.MODEL_TENSOR.ATTN_OUT,
            gguf.MODEL_TENSOR.FFN_UP,
            gguf.MODEL_TENSOR.FFN_DOWN,
            gguf.MODEL_TENSOR.FFN_GATE,
        ]):
            # transform weight into 1/0/-1 (in fp32)
            data_torch = self.weight_quant(data_torch)

        yield from super().modify_tensors(data_torch, name, bid)
