# conversion.base.load_hf_tokenizer on a byte-level BPE tokenizer.json whose tokenizer_config.json names LlamaTokenizerFast
# (the DeepSeek LLM / V3 / R1 layout). transformers 5.x LlamaTokenizer replaces the ByteLevel pre-tokenizer with Metaspace
# (huggingface/transformers#45488), dropping every space; load_hf_tokenizer must hand back what transformers 4.x gave:
# the tokenizer.json pieces, with BOS prepended when add_bos_token is true (LlamaTokenizerFast.update_post_processor).
# The reference is computed from tokenizer.json with the tokenizers library alone, so it does not depend on transformers.
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")  # conversion.base imports torch at module level
pytest.importorskip("transformers")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

from conversion.base import load_hf_tokenizer  # noqa: E402

TEXTS = ["Hi there", "hello world, how are you?", "  two  spaces\nand a newline"]


def make_dir(d: Path, cfg: dict) -> None:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<s>", "</s>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(TEXTS * 20, trainer=trainer)
    d.mkdir(parents=True, exist_ok=True)
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")


def reference(d: Path, add_bos: bool) -> list[list[int]]:
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    bos = tok.token_to_id("<s>")
    return [([bos] if add_bos else []) + tok.encode(t, add_special_tokens=False).ids for t in TEXTS]


@pytest.mark.parametrize("add_bos", [True, False, None])
def test_byte_level_llama_config(add_bos):
    cfg = {"tokenizer_class": "LlamaTokenizerFast", "bos_token": "<s>", "eos_token": "</s>", "add_eos_token": False}
    if add_bos is not None:
        cfg["add_bos_token"] = add_bos
    with tempfile.TemporaryDirectory(prefix="bl_llama_") as td:
        d = Path(td)
        make_dir(d, cfg)
        tok = load_hf_tokenizer(d)
        got = [tok.encode(t) for t in TEXTS]
        assert got == reference(d, add_bos is not False)  # LlamaTokenizerFast defaults add_bos_token to true
        for t in TEXTS:
            assert tok.decode(tok.encode(t, add_special_tokens=False)) == t


def test_byte_level_without_llama_class_untouched():
    # control: PreTrainedTokenizerFast already keeps tokenizer.json; add_bos_token is not consulted (no BOS template)
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>", "eos_token": "</s>", "add_bos_token": True}
    with tempfile.TemporaryDirectory(prefix="bl_ptf_") as td:
        d = Path(td)
        make_dir(d, cfg)
        tok = load_hf_tokenizer(d)
        assert [tok.encode(t) for t in TEXTS] == reference(d, False)
