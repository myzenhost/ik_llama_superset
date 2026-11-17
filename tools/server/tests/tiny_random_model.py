# Random-weight 2-layer `llama` GGUF for server tests that need no downloaded model. Prompts are sent as token-id arrays,
# so the vocabulary only has to exist (it is never decoded to text by these tests).
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "gguf-py"))
import gguf  # noqa: E402

V = 64
E = 32


def write_tiny_llama(path: str, seed: int = 2468, n_layer: int = 2, n_ctx: int = 1024) -> str:
    rng = np.random.default_rng(seed)

    def rnd(*shape):
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    w = gguf.GGUFWriter(path, "llama")
    w.add_name("tiny-random")
    w.add_context_length(n_ctx)
    w.add_embedding_length(E)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(2 * E)
    w.add_head_count(4)
    w.add_head_count_kv(4)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(E // 4)
    # single-character tokens for a-z, '"' and newline (so grammar-constrained tests can spell their strings), then filler
    chars = list("abcdefghijklmnopqrstuvwxyz") + ['"']
    toks = ["<pad>", "<s>", "</s>", "Ċ"] + chars + ["t%d" % i for i in range(4 + len(chars), V - 1)] + ["ab"]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(toks)
    w.add_token_types([gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.NORMAL] * (V - 3))
    w.add_token_merges(["a b"])
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_pad_token_id(0)
    w.add_tensor("token_embd.weight", rnd(V, E))
    for i in range(n_layer):
        w.add_tensor(f"blk.{i}.attn_norm.weight", 1.0 + rnd(E))
        for nm in ("attn_q", "attn_k", "attn_v", "attn_output"):
            w.add_tensor(f"blk.{i}.{nm}.weight", rnd(E, E))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", 1.0 + rnd(E))
        w.add_tensor(f"blk.{i}.ffn_gate.weight", rnd(2 * E, E))
        w.add_tensor(f"blk.{i}.ffn_up.weight", rnd(2 * E, E))
        w.add_tensor(f"blk.{i}.ffn_down.weight", rnd(E, 2 * E))
    w.add_tensor("output_norm.weight", 1.0 + rnd(E))
    w.add_tensor("output.weight", rnd(V, E))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def tiny_server(tmp_path):
    """A ServerProcess on a freshly written random-weight model (no network, no HF cache)."""
    from utils import ServerProcess
    server = ServerProcess()
    server.offline = True
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = write_tiny_llama(str(tmp_path / "tiny.gguf"))
    server.model_alias = "tiny-random"
    server.n_ctx = 1024
    server.n_batch = 64
    server.n_slots = 1
    server.n_predict = 4
    server.seed = 42
    return server


PROMPT = [1] + [(i * 7) % 50 + 4 for i in range(80)]
