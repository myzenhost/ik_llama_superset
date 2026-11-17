from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("GPTJForCausalLM")
@ModelBase.example("EleutherAI/gpt-j-6b")
class GPTJModel(TextModel):
    """GPT-J: parallel attention + MLP behind ONE LayerNorm (ln_1), bias-free q/k/v/out_proj, biased fc_in/fc_out and lm_head,
    interleaved ("rotate_every_two") RoPE on the first rotary_dim channels of each head.
    Reference: transformers models/gptj/modeling_gptj.py (GPTJBlock, GPTJAttention, apply_rotary_pos_emb)."""
    model_arch = gguf.MODEL_ARCH.GPTJ

    def set_gguf_parameters(self):
        hp = self.hparams
        n_embd = self.find_hparam(["n_embd", "hidden_size"])
        n_head = self.find_hparam(["n_head", "num_attention_heads"])
        rotary_dim = hp.get("rotary_dim")
        if rotary_dim is None:
            # modeling_gptj.py:118-120,221,233: rotary_dim=None rotates the whole head
            rotary_dim = n_embd // n_head
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(self.find_hparam(["n_positions", "max_position_embeddings", "n_ctx"]))
        self.gguf_writer.add_embedding_length(n_embd)
        # modeling_gptj.py:431: inner_dim = n_inner if n_inner is not None else 4 * n_embd
        n_inner = hp.get("n_inner")
        self.gguf_writer.add_feed_forward_length(n_inner if n_inner is not None else 4 * n_embd)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_rope_dimension_count(rotary_dim)
        # modeling_gptj.py:63 create_sinusoidal_positions: base 10000, not configurable
        self.gguf_writer.add_rope_freq_base(10000.0)
        self.gguf_writer.add_layer_norm_eps(hp.get("layer_norm_epsilon", 1e-5))
        self.gguf_writer.add_file_type(self.ftype)
        act = hp.get("activation_function", "gelu_new")
        if act != "gelu_new":
            # the graph is the tanh GELU (ACT2FN["gelu_new"]); refuse to silently convert a different activation
            raise ValueError(f"GPT-J activation_function {act!r} is not implemented (only gelu_new)")

    def get_vocab_base_pre(self, tokenizer) -> str:
        # Every GPT-J tokenizer is GPT2Tokenizer, whose pre-tokenizer is the fixed GPT-2 regex. The chkhsh table only knows the
        # vocabularies of published checkpoints, so a retrained or tiny-random vocab (hf-internal-testing/tiny-random-GPTJForCausalLM,
        # Milos/slovak-gpt-j-405M) raised "BPE pre-tokenizer was not recognized". Fall back to the GPT-2 regex; the parity test
        # (temp/gptj_1107/gptj_parity.py) compares llama.cpp token ids against the HF tokenizer's, so a wrong fallback shows there.
        try:
            return super().get_vocab_base_pre(tokenizer)
        except NotImplementedError:
            logger.warning("GPT-J: unrecognised BPE vocabulary hash, using the GPT-2 pre-tokenizer regex (pre = 'gpt-2')")
            return "gpt-2"

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # causal-mask buffers stored by old checkpoints, not weights
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            return
        if name == "lm_head.weight" and self.hparams.get("tie_word_embeddings", False):
            # tied checkpoint: token_embd doubles as output; the C++ loader requires an explicit output tensor, so keep it
            logger.info("GPT-J tie_word_embeddings=True: writing output.weight from lm_head.weight")
        yield from super().modify_tensors(data_torch, name, bid)
