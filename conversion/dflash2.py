"""DFlash 2 draft model (dynamic convolution + candidate selector) -- ported from ik_llama.cpp's
convert_hf_to_gguf.py class DFlash2DraftModel (ik-main merge 2026-09-29, follow-up 2026-09-30).

PROGRESS-OK: no long process here. `modify_tensors` is a per-tensor CALLBACK; progress and ETA belong
to the outer loop in conversion/base.py.

RESEARCH-DONE: ik source read in full (class DFlash2DraftModel, 151 lines, in the ik-main diff dumped
to temp/_ikdiff_conv_0930.txt lines 81-230). ik_llama.cpp is MIT; this is a re-expression against the
fork's conversion/ package, not a text patch: ik's converter is one file whose DFlashDraftModel base
carries helpers (`_require_target_model_dir`, `_get_target_hparams`, `_get_target_raw_hparams`,
`_causal_attention`) that the package's DFlashModel does not define. They are carried here, as
conversion/dflash_laguna.py does for the same reason.

RUNTIME SIDE ALREADY PRESENT IN THIS FORK (checked before writing): LLM_ARCH_DFLASH2 in
src/llama-arch.cpp:98 (arch string "dflash"), hparams read at src/llama-hparams.cpp:2308 (keys
dflash.block_size / conv_kernel_size / conv_group_size / selector_rank / selector_top_k,
tokenizer mask id, target_layers one-based), tensor loader at src/llama-load-tensors.cpp:6267, and
the gguf-py constants, tensor names and HF->GGUF mappings for the conv and selector tensors. The
converter was the only missing half, so a DFlash-2 checkpoint could not be turned into a GGUF.

Raw GGUF names "fc.weight" and "enc.output_norm.weight" are what src/llama-model.cpp:1262-1263 loads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .qwen import DFlashModel, Qwen3Model


@ModelBase.register("DFlash2DraftModel")
class DFlash2Model(DFlashModel):
    """DFlash 2 sidecar with dynamic convolution and candidate selector tensors."""

    model_arch = gguf.MODEL_ARCH.DFLASH2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_raw_hparams: dict[str, Any] | None = None

    # ---- helpers carried from ik's DFlashDraftModel ---------------------------------------------

    def _require_target_model_dir(self) -> Path:
        if self.target_model_dir is None:
            raise ValueError("DFlash2DraftModel conversion requires "
                             "--target-model-dir <matching target model directory>")
        return self.target_model_dir

    def _get_target_raw_hparams(self) -> dict[str, Any]:
        if self._target_raw_hparams is None:
            with open(self._require_target_model_dir() / "config.json", "r", encoding="utf-8") as f:
                self._target_raw_hparams = json.load(f)
        return self._target_raw_hparams

    def _get_target_hparams(self) -> dict[str, Any]:
        # text hparams: a multimodal target nests them under text_config
        raw = self._get_target_raw_hparams()
        text = raw.get("text_config")
        return {**raw, **text} if isinstance(text, dict) else raw

    def _causal_attention(self) -> bool:
        return False

    # ---- vocab ------------------------------------------------------------------------------------

    def _set_vocab_tokenizer_json(self, dir_model: Path, vocab_size: int) -> None:
        tokenizer_path = dir_model / "tokenizer.json"
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)

        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        class TokenizerShim:
            def encode(self, text: str) -> list[int]:
                return tokenizer.encode(text).ids

        vocab: dict[str, int] = tokenizer_json["model"]["vocab"]
        reverse_vocab = {id_: token for token, id_ in vocab.items()}
        added_vocab = {
            item["id"]: item
            for item in tokenizer_json.get("added_tokens", [])
            if isinstance(item.get("id"), int) and isinstance(item.get("content"), str)
        }
        reverse_vocab.update({id_: item["content"] for id_, item in added_vocab.items()})
        assert max(reverse_vocab) < vocab_size

        tokpre = self.get_vocab_base_pre(TokenizerShim())
        tokens: list[str] = []
        toktypes: list[int] = []
        for i in range(vocab_size):
            token = reverse_vocab.get(i)
            if token is None:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
                continue

            added_token = added_vocab.get(i)
            if added_token is not None:
                if not added_token.get("normalized", True):
                    token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False).ids)
                if added_token.get("special", False) or self.does_token_look_special(token):
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    token = token.replace("▁", " ")
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                toktypes.append(gguf.TokenType.NORMAL)
            tokens.append(token)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)
        eot_id = next(
            (id_ for id_, item in added_vocab.items() if item["content"] == "<|eot|>"),
            None,
        )
        if eot_id is not None:
            self.gguf_writer.add_eot_token_id(eot_id)

    def set_vocab(self):
        target_dir = self._require_target_model_dir()
        target_hparams = self._get_target_hparams()
        target_raw_hparams = self._get_target_raw_hparams()
        target_architectures = target_raw_hparams.get("architectures", [])
        if "MuseGlimmerForConditionalGeneration" in target_architectures:
            self._set_vocab_tokenizer_json(target_dir, int(target_hparams["vocab_size"]))
        else:
            super().set_vocab()
            if (bos_token_id := target_hparams.get("bos_token_id")) is not None:
                self.gguf_writer.add_bos_token_id(int(bos_token_id))

    # ---- hyper-parameters -------------------------------------------------------------------------

    def set_gguf_parameters(self):
        dflash_cfg = self.hparams.get("dflash_config")
        dflash_cfg = dflash_cfg if isinstance(dflash_cfg, dict) else {}

        # Skip DFlashModel.set_gguf_parameters: it writes block_size/target_layers with DFlash-1
        # defaults (block_size 16 when absent), where DFlash 2 must refuse a checkpoint that omits them.
        Qwen3Model.set_gguf_parameters(self)
        self.gguf_writer.add_causal_attention(self._causal_attention())

        rope_parameters = self.hparams.get("rope_parameters")
        if isinstance(rope_parameters, dict) and (rope_theta := rope_parameters.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(float(rope_theta))

        def required(name: str) -> int:
            value = dflash_cfg.get(name, self.hparams.get(name))
            if value is None:
                raise ValueError(f"DFlash2DraftModel conversion requires explicit {name} metadata")
            return int(value)

        block_size = required("block_size")
        mask_token_id = required("mask_token_id")
        target_layer_ids = dflash_cfg.get("target_layer_ids")
        if target_layer_ids is None:
            raise ValueError("DFlash2DraftModel conversion requires target_layer_ids metadata")
        target_layers = [int(value) + 1 for value in target_layer_ids]
        if not target_layers or any(value <= 0 for value in target_layers):
            raise ValueError("DFlash2DraftModel conversion requires target_layer_ids metadata")
        if len(set(target_layers)) != len(target_layers):
            raise ValueError("DFlash2DraftModel conversion requires unique target_layer_ids metadata")

        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.block_size", block_size)
        self.gguf_writer.add_mask_token_id(mask_token_id)
        self.gguf_writer.add_array(f"{self.gguf_writer.arch}.target_layers", target_layers)
        self.gguf_writer.add_conv_kernel_size(required("conv_kernel_size"))
        self.gguf_writer.add_conv_group_size(required("conv_group_size"))
        self.gguf_writer.add_selector_rank(required("selector_rank"))
        self.gguf_writer.add_selector_top_k(required("selector_top_k"))

        for name, method in (
            ("output_multiplier", self.gguf_writer.add_logit_scale),
            ("final_logit_softcapping", self.gguf_writer.add_final_logit_softcapping),
            ("input_embedding_scale", self.gguf_writer.add_embedding_scale),
        ):
            value = dflash_cfg.get(name, self.hparams.get(name))
            if value is not None and (name != "final_logit_softcapping" or float(value) > 0):
                method(float(value))

        use_sliding_window = self.hparams.get("use_sliding_window")
        sliding_window = self.hparams.get("sliding_window")
        if use_sliding_window and sliding_window:
            layer_types = self.hparams.get("layer_types")
            swa_pattern = ([str(value) == "sliding_attention" for value in layer_types]
                           if layer_types else [True] * self.block_count)
            self.gguf_writer.add_sliding_window(int(sliding_window))
            self.gguf_writer.add_sliding_window_pattern(swa_pattern)

    # ---- tensors ----------------------------------------------------------------------------------

    @staticmethod
    def normalize_tensor_name(name: str) -> str:
        if name.startswith("candidate_selector."):
            name = f"model.{name}"
        if name in (
            "model.candidate_selector.predecessor_codebook",
            "model.candidate_selector.successor_codebook",
        ):
            name += ".weight"
        return name

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        top_level_name = name[6:] if name.startswith("model.") else name
        if top_level_name == "fc.weight":
            return [("fc.weight", data_torch)]
        if top_level_name == "hidden_norm.weight":
            return [("enc.output_norm.weight", data_torch)]
        return super().modify_tensors(data_torch, self.normalize_tensor_name(name), bid)
