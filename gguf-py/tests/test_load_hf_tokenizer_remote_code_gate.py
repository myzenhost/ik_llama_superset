# The converter must not import and run a model repo's own Python unless asked to (--trust-remote-code, which sets
# conversion.base.ALLOW_REMOTE_CODE). Call sites pass trust_remote_code=True for architectures whose tokenizer may be repo
# code; before this gate that alone ran it. Each fixture module writes a marker file when it is executed, so "did it run"
# is measured, not inferred from whether loading succeeded.
# load_kimi_tiktoken builds the Kimi tokenizer from tiktoken.model plus two literals read from tokenization_kimi.py's
# syntax tree; the same marker shows that file is never executed.
import base64
import json
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")  # conversion.base imports torch at module level
pytest.importorskip("transformers")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

import conversion.base as cb  # noqa: E402

TEXTS = ["Hi there", "hello world, how are you?"]


def make_auto_map_repo(d: Path, marker: Path) -> None:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<s>", "</s>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(TEXTS * 20, trainer=trainer)
    d.mkdir(parents=True, exist_ok=True)
    tok.save(str(d / "tokenizer.json"))
    mod = "gate_tok_" + uuid.uuid4().hex[:12]  # a fresh module name: transformers caches dynamic modules by name
    (d / (mod + ".py")).write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('ran')\n"
        "from transformers import PreTrainedTokenizerFast\n"
        "class GateTokenizer(PreTrainedTokenizerFast):\n"
        "    pass\n", encoding="utf-8")
    cfg = {"tokenizer_class": "GateTokenizer", "bos_token": "<s>", "eos_token": "</s>",
           "auto_map": {"AutoTokenizer": [None, mod + ".GateTokenizer"]}}
    (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")


def test_remote_code_refused_without_flag(monkeypatch):
    monkeypatch.setattr(cb, "ALLOW_REMOTE_CODE", False, raising=False)
    with tempfile.TemporaryDirectory(prefix="rc_gate_") as td:
        marker = Path(td) / "MARKER"
        make_auto_map_repo(Path(td) / "repo", marker)
        with pytest.raises(ValueError, match="--trust-remote-code"):
            cb.load_hf_tokenizer(Path(td) / "repo", trust_remote_code=True)
        assert not marker.exists(), "the repo's tokenizer module ran without --trust-remote-code"


def test_remote_code_runs_with_flag(monkeypatch):
    # control: the fixture really is code that runs when allowed, so the refusal above is not vacuous
    monkeypatch.setattr(cb, "ALLOW_REMOTE_CODE", True, raising=False)
    with tempfile.TemporaryDirectory(prefix="rc_allow_") as td:
        marker = Path(td) / "MARKER"
        make_auto_map_repo(Path(td) / "repo", marker)
        tok = cb.load_hf_tokenizer(Path(td) / "repo", trust_remote_code=True)
        assert marker.exists()
        assert type(tok).__name__ == "GateTokenizer"


PAT = r""" ?\p{L}+| ?[^\s\p{L}\p{N}]+|\p{N}{1,3}|\s+(?!\S)|\s+"""
MERGES = [b"th", b"he", b"the", b"ll", b"llo", b" t", b" the", b"er", b"ere"]


def make_kimi_repo(d: Path, marker: Path, literal_pat: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    ranks = [bytes([b]) for b in range(256)] + MERGES
    (d / "tiktoken.model").write_text("".join(f"{base64.b64encode(t).decode()} {i}\n" for i, t in enumerate(ranks)), encoding="utf-8")
    parts = PAT.split("|")
    pat_src = ("    pat_str = \"|\".join([" + ", ".join(f"r\"\"\"{p}\"\"\"" for p in parts) + "])\n") if literal_pat else \
              "    pat_str = PATTERN\n"
    (d / "tokenization_kimi.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('ran')\n"
        "class TikTokenTokenizer:\n"
        "    num_reserved_special_tokens = 4\n" + pat_src, encoding="utf-8")
    n = len(ranks)
    cfg = {"bos_token": "[BOS]", "eos_token": "[EOS]", "tokenizer_class": "TikTokenTokenizer",
           "added_tokens_decoder": {str(n): {"content": "[BOS]"}, str(n + 1): {"content": "[EOS]"}, str(n + 5): {"content": "[PAD]"}},
           "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", None]}}
    (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")


def test_kimi_tiktoken_without_running_repo_code():
    tiktoken = pytest.importorskip("tiktoken")
    from tiktoken.load import load_tiktoken_bpe
    with tempfile.TemporaryDirectory(prefix="kimi_tt_") as td:
        marker = Path(td) / "MARKER"
        d = Path(td) / "repo"
        make_kimi_repo(d, marker)
        tok = cb.load_kimi_tiktoken(d)
        assert tok is not None
        assert not marker.exists(), "tokenization_kimi.py was executed"
        n = 256 + len(MERGES)
        # ids n .. n+4+1 as the class builds them: named from added_tokens_decoder, else <|reserved_token_i|>
        assert tok.special_tokens == {"[BOS]": n, "[EOS]": n + 1, "<|reserved_token_%d|>" % (n + 2): n + 2,
                                      "<|reserved_token_%d|>" % (n + 3): n + 3, "<|reserved_token_%d|>" % (n + 4): n + 4, "[PAD]": n + 5}
        assert tok.eos_id == n + 1
        assert tok.model._mergeable_ranks == load_tiktoken_bpe(str(d / "tiktoken.model"))
        ref = tiktoken.Encoding(name="ref", pat_str=PAT, mergeable_ranks=load_tiktoken_bpe(str(d / "tiktoken.model")), special_tokens={})
        for t in TEXTS + ["the there 123456 [EOS] <|x|>"]:
            assert tok.encode(t) == ref.encode(t, disallowed_special=())
        assert tok.encode(" the")[0] == 256 + MERGES.index(b" the")  # the merges are in effect, not byte-by-byte


def test_kimi_tiktoken_declines_non_literal_pattern():
    pytest.importorskip("tiktoken")
    with tempfile.TemporaryDirectory(prefix="kimi_nl_") as td:
        marker = Path(td) / "MARKER"
        make_kimi_repo(Path(td) / "repo", marker, literal_pat=False)
        assert cb.load_kimi_tiktoken(Path(td) / "repo") is None
        assert not marker.exists()


def test_kimi_tiktoken_not_a_kimi_dir():
    with tempfile.TemporaryDirectory(prefix="kimi_none_") as td:
        assert cb.load_kimi_tiktoken(td) is None
