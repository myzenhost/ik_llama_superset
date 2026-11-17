"""DiffusionGemma -- block text-diffusion MoE on a Gemma-4 backbone.

PROGRESS-OK: no long process here. `modify_tensors` is a per-tensor CALLBACK; the conversion run's
progress, ETA and heartbeat belong to the outer loop in conversion/base.py, which owns it.

RESEARCH-DONE: this class is being RETURNED to the file it came from. The fork's own copy, at
HEAD:convert_hf_to_gguf.py lines 4089-4189 (5,511 bytes), carries this note verbatim:
    "Adapted from unsloth's `conversion/diffusion_gemma.py`. That tree splits the converter into a
     `conversion/` package of per-arch modules; this fork has a single monolithic
     convert_hf_to_gguf.py, so the class lives here and hooks Gemma4Model rather than subclassing
     a package module."
The merge brings that package structure in, so the class goes back to being a package module.
NOTHING IS DISABLED. The `return []` branches are this model's CORRECT conversion semantics,
unchanged: the encoder shares ALL weights with the decoder except a per-layer layer_scalar, so
emitting the other encoder tensors would produce a file that loads and denoises with a DUPLICATED
stack; and the vision tower is not part of the text-diffusion path. Both are stated in the original.

WHY IT EXISTS AS A SEPARATE FILE. Adopting upstream's conversion/ package gains 194 architectures
and drops 6 unless they are carried (fork registry 85, package 273, shared 79; controls: the sets
overlap by 79, an invented name is in neither). DiffusionGemma4ModelForBlockDiffusion and
DiffusionGemmaForBlockDiffusion are two of the 6.

TWO ADAPTATIONS, each verified rather than assumed:
  * `@Model.register(...)` -> `@ModelBase.register(...)` (the package's registry decorator).
  * `self._text_hparams()` -> `self.hparams`. `_text_hparams` is a fork-only helper and exists
    NOWHERE in the package (0 hits across conversion/*.py and gguf-py/gguf/*.py, against
    `hparams` = 70 in gemma.py as the control). The package needs no such helper because
    conversion/base.py:1186-1188 already hoists it:
        if "text_config" in self.hparams:
            # move the text_config to the root level
            self.hparams = {**self.hparams, **self.hparams["text_config"]}
    so self.hparams IS the text hparams here, and the setdefault calls below still seed the
    parent's set_gguf_parameters exactly as before.

EVERY OTHER SYMBOL VERIFIED PRESENT FIRST, word-bounded, control DIFFUSION_GEMMA = 3:
    staged constants.py : DIFFUSION_GEMMA 3 · ENC_LAYER_OUT_SCALE 3 · SC_PRE_NORM 3 · SC_GATE 3 ·
                          SC_UP 3 · SC_DOWN 3
    staged gguf_writer  : add_diffusion_canvas_length 1 · add_diffusion_eb_max_steps 1 ·
                          add_diffusion_eb_t_min 1 · add_diffusion_eb_t_max 1 (line 552) ·
                          add_diffusion_eb_entropy_bound 1 ·
                          add_diffusion_eb_stability_threshold 1 ·
                          add_diffusion_eb_confidence_threshold 1 · add_causal_attention 1
    conversion/base.py  : find_hparam 19 · format_tensor_name 2 · dir_model 55
    conversion/gemma.py : class Gemma4Model(Gemma3Model) at line 633

NOTE: the decorator alone does NOT make these models reachable. conversion/__init__.py's
TEXT_MODEL_MAP is the dispatch and needs an entry per architecture name.
"""
from __future__ import annotations

import json
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .gemma import Gemma4Model


@ModelBase.register("DiffusionGemma4ModelForBlockDiffusion", "DiffusionGemmaForBlockDiffusion")
class DiffusionGemmaModel(Gemma4Model):
    """Block text-diffusion MoE on a Gemma-4 backbone.

    Encoder (causal prefill) and decoder (bidirectional canvas denoising) share ALL weights except
    a per-layer layer_scalar, and the backbone lives under `model.decoder.*`. Strategy: rewrite
    `model.decoder.<x>` -> `model.<x>` so the inherited Gemma4 tensor map handles it, then export
    the encoder layer_scalars (ENC_LAYER_OUT_SCALE) and the self_conditioning gated MLP (SC_*)
    explicitly, and pass over every other encoder tensor -- they are TIED to the decoder, so
    emitting them would produce a file that loads and denoises with a duplicated stack.

    The vision tower is not part of the text-diffusion path; lm_head is tied to the decoder
    embeddings.
    """

    model_arch = gguf.MODEL_ARCH.DIFFUSION_GEMMA

    def set_gguf_parameters(self):
        # Plain Gemma-4 MoE. DiffusionGemma checkpoints carry neither the gemma3n per-layer-input
        # embeddings nor shared-KV layers, so default those two hparams to 0 for the parent.
        # PORT NOTE: the fork read these from its own _text_hparams(); in this package
        # conversion/base.py has already hoisted text_config to the root of self.hparams.
        hparams = self.hparams
        hparams.setdefault("num_kv_shared_layers", 0)
        hparams.setdefault("hidden_size_per_layer_input", 0)

        super().set_gguf_parameters()

        # bidirectional decoder; the forward fills its own region-aware mask
        self.gguf_writer.add_causal_attention(False)

        # canvas_length is REQUIRED -- the runtime splits [prompt | canvas] on it, and a zero would
        # make the whole batch "prompt" and denoise nothing.
        canvas_length = self.find_hparam(["canvas_length"], optional=False)
        if canvas_length is None or int(canvas_length) <= 0:
            raise ValueError("DiffusionGemma conversion requires a positive root canvas_length")
        self.gguf_writer.add_diffusion_canvas_length(int(canvas_length))

        # entropy-bound sampler defaults (the reference decoder) from generation_config; every key
        # is optional so older configs still convert.
        gen_cfg_path = self.dir_model / "generation_config.json"
        if gen_cfg_path.is_file():
            with open(gen_cfg_path, encoding="utf-8") as f:
                gen_cfg = json.load(f)
            sampler_cfg = gen_cfg.get("sampler_config", {})
            if "max_denoising_steps" in gen_cfg:
                self.gguf_writer.add_diffusion_eb_max_steps(int(gen_cfg["max_denoising_steps"]))
            if "t_min" in gen_cfg:
                self.gguf_writer.add_diffusion_eb_t_min(float(gen_cfg["t_min"]))
            if "t_max" in gen_cfg:
                self.gguf_writer.add_diffusion_eb_t_max(float(gen_cfg["t_max"]))
            if "entropy_bound" in sampler_cfg:
                self.gguf_writer.add_diffusion_eb_entropy_bound(
                    float(sampler_cfg["entropy_bound"]))
            if "stability_threshold" in gen_cfg:
                self.gguf_writer.add_diffusion_eb_stability_threshold(
                    int(gen_cfg["stability_threshold"]))
            if "confidence_threshold" in gen_cfg:
                self.gguf_writer.add_diffusion_eb_confidence_threshold(
                    float(gen_cfg["confidence_threshold"]))

    def modify_tensors(self, data_torch: Tensor, name: str,
                       bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # raw 1-D scalar buffers arrive without a ".weight" suffix
        if name.endswith("layer_scalar"):
            name = name + ".weight"

        # the converter passes tensors through a "language_model." prefix; DiffusionGemma
        # checkpoints do not carry it, so strip-then-reattach around our own rewrites.
        stripped = name.replace("language_model.", "")

        # the vision tower is not part of the text-diffusion path
        if "vision" in stripped or "embed_vision" in stripped:
            return []

        # encoder-mode per-layer scalar -> its own ENC_LAYER_OUT_SCALE tensor
        if stripped.startswith("model.encoder.layers.") and "layer_scalar" in stripped:
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENC_LAYER_OUT_SCALE, bid),
                     data_torch)]

        # every OTHER encoder tensor is TIED to the decoder -- emitting it would duplicate the
        # stack, so the decoder's copy is the one that is written
        if stripped.startswith("model.encoder."):
            return []

        # decoder-only self-conditioning gated MLP
        if stripped.startswith("model.decoder.self_conditioning."):
            sub = stripped[len("model.decoder.self_conditioning."):]
            sc_map = {
                "pre_norm.weight":  gguf.MODEL_TENSOR.SC_PRE_NORM,
                "gate_proj.weight": gguf.MODEL_TENSOR.SC_GATE,
                "up_proj.weight":   gguf.MODEL_TENSOR.SC_UP,
                "down_proj.weight": gguf.MODEL_TENSOR.SC_DOWN,
            }
            if sub in sc_map:
                return [(self.format_tensor_name(sc_map[sub]), data_torch)]
            return []

        # remap the backbone (everything else under model.decoder.*) to model.<x> for Gemma4Model
        if stripped.startswith("model.decoder."):
            stripped = "model." + stripped[len("model.decoder."):]

        return super().modify_tensors(data_torch, "language_model." + stripped, bid)
