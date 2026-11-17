"""SpecialVocab: tokenizer_config add_bos_token / add_eos_token only reach the GGUF when transformers would act on them.
PROGRESS-OK: a few small JSON files per test, milliseconds.

transformers' generic fast tokenizer (PreTrainedTokenizerFast in 4.x, TokenizersBackend in 5.x) encodes with the
tokenizer.json post_processor and ignores those keys; only Llama / CodeLlama / Gemma / Cohere / GPT-NeoX rebuild the
post_processor from them. SpecialVocab used to copy the keys regardless, so a hunyuan-dense GGUF prepended a BOS that HF
never adds, and minicpm5 / afmoe GGUFs dropped the BOS that HF does add.

The REAL_DIRS expectations were measured on 2026-10-15 with transformers 4.57.6 and 5.20.0 (both agree for these four
except deepseek-v3, where 5.20 no longer honours the Llama flag; the expectation follows 4.57 and the model's own
converter, which asserts add_bos_token true).
Run:  python -m pytest gguf-py/tests/test_special_vocab_add_flags.py
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GGUF_PY = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, GGUF_PY)
ROOT = os.path.normpath(os.path.join(GGUF_PY, ".."))

from gguf.vocab import SpecialVocab  # noqa: E402

BOS_TEMPLATE = {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                         {"SpecialToken": {"id": "<s>", "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}}],
                "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}}
NO_BOS = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}


def _vocab(tmp_path, post_processor, config, with_tokenizer_json=True):
    if with_tokenizer_json:
        tok = {"model": {"type": "BPE", "vocab": {}, "merges": []}, "post_processor": post_processor,
               "added_tokens": [{"id": 1, "content": "<s>"}, {"id": 2, "content": "</s>"}]}
        (tmp_path / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    cfg = {"bos_token": "<s>", "eos_token": "</s>", **config}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return SpecialVocab(tmp_path, load_merges=False).add_special_token


def test_generic_class_true_flag_does_not_add_a_bos_the_post_processor_lacks(tmp_path):
    add = _vocab(tmp_path, NO_BOS, {"tokenizer_class": "PreTrainedTokenizerFast", "add_bos_token": True})
    assert add.get("bos") is not True


def test_generic_class_false_flag_does_not_drop_the_post_processor_bos(tmp_path):
    add = _vocab(tmp_path, BOS_TEMPLATE, {"tokenizer_class": "PreTrainedTokenizerFast", "add_bos_token": False})
    assert add.get("bos") is True


def test_generic_class_eos_flag_ignored_too(tmp_path):
    add = _vocab(tmp_path, NO_BOS, {"tokenizer_class": "PreTrainedTokenizerFast", "add_eos_token": True})
    assert add.get("eos") is not True


@pytest.mark.parametrize("cls", ["LlamaTokenizerFast", "LlamaTokenizer", "GemmaTokenizerFast", "CodeLlamaTokenizerFast",
                                 "CohereTokenizerFast", "GPTNeoXTokenizerFast"])
def test_classes_that_rebuild_the_post_processor_keep_the_flag(tmp_path, cls):
    assert _vocab(tmp_path, NO_BOS, {"tokenizer_class": cls, "add_bos_token": True}).get("bos") is True
    assert _vocab(tmp_path, BOS_TEMPLATE, {"tokenizer_class": cls, "add_bos_token": False}).get("bos") is False


def test_no_tokenizer_json_keeps_the_flag(tmp_path):
    add = _vocab(tmp_path, None, {"tokenizer_class": "PreTrainedTokenizerFast", "add_bos_token": True}, with_tokenizer_json=False)
    assert add.get("bos") is True


def test_custom_tokenizer_code_keeps_the_flag(tmp_path):
    add = _vocab(tmp_path, NO_BOS, {"tokenizer_class": "XTokenizer", "add_bos_token": True,
                                    "auto_map": {"AutoTokenizer": ["tok.XTokenizer", None]}})
    assert add.get("bos") is True


# name -> does HF raw encode start with the BOS (measured, see module docstring)
REAL_DIRS = {"hunyuan-dense": False, "minicpm5": True, "afmoe": True, "deepseek-v3": True}


@pytest.mark.parametrize("name,hf_adds_bos", sorted(REAL_DIRS.items()))
def test_real_tokenizer_dirs_match_measured_hf(name, hf_adds_bos):
    d = os.path.join(ROOT, "models", "tokenizers", name)
    if not os.path.isfile(os.path.join(d, "tokenizer.json")):
        pytest.skip("tokenizer dir not present: " + d)
    from pathlib import Path
    add = SpecialVocab(Path(d), load_merges=False).add_special_token
    assert bool(add.get("bos", False)) is hf_adds_bos, (name, add)
