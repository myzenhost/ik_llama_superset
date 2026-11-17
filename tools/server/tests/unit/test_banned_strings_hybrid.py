"""Banned strings on FULL seq_rm models (recurrent / hybrid: lfm2, granitehybrid, mamba ...).

A banned-string rewind must undo tokens already decoded. Attention-only memory does that with a
partial seq_rm; recurrent state cannot be cut back, so the server keeps one PARTIAL_ONLY state
checkpoint per buffered token and restores the one taken before the matched text started.
Without it the server can only switch bans off for these models, and the banned text is emitted.

The ggml-org presets carry no hybrid model, so this module runs against a LOCAL GGUF named by
LLAMA_TEST_HYBRID_MODEL (os.pathsep-separated) and is skipped without it -- as test_hybrid_state.py.
Every test bans something the greedy reference DOES contain, so a server that ignores the ban fails.
"""
import os
import re

import pytest
from utils import *

MODELS = [p for p in os.environ.get("LLAMA_TEST_HYBRID_MODEL", "").split(os.pathsep) if p]
pytestmark = pytest.mark.skipif(not MODELS, reason="set LLAMA_TEST_HYBRID_MODEL to a local hybrid GGUF")

PROMPT = "Once upon a time, in a small village by the sea,"
N_PREDICT = 48


def _server(model: str) -> ServerProcess:
    s = ServerProcess()
    s.offline = True
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.n_ctx = 1024
    s.n_slots = 1
    s.seed = 1
    return s


def _complete(s: ServerProcess, **extra) -> dict:
    res = s.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": N_PREDICT, "temperature": 0, "cache_prompt": False, **extra,
    })
    assert res.status_code == 200, res.body
    return res.body


def _a_word(text: str, nth: int = 1) -> str:
    words = re.findall(r"[A-Za-z]{4,}", text)
    assert len(words) > nth, text
    return words[nth]


@pytest.fixture(params=MODELS, ids=[os.path.basename(m) for m in MODELS])
def server(request):
    s = _server(request.param)
    s.start()
    yield s
    s.stop()


@pytest.mark.parametrize("nth", [0, 1, 3])
def test_banned_string_is_not_emitted_and_the_prefix_is_kept(server, nth):
    ref = _complete(server)["content"]
    word = _a_word(ref, nth)
    at = ref.lower().find(word.lower())

    got = _complete(server, banned_strings=[word])["content"]
    assert word.lower() not in got.lower(), (word, got)
    # the restored state must be the state BEFORE the match: the text up to it is unchanged
    assert got[:at] == ref[:at], (at, ref, got)


def test_banned_regex(server):
    ref = _complete(server)["content"]
    word = _a_word(ref)
    pattern = re.escape(word[:3]) + "[a-z]+"
    assert re.search(pattern, ref)
    got = _complete(server, banned_regex=[pattern])["content"]
    assert not re.search(pattern, got), (pattern, got)


def test_no_bans_is_unchanged(server):
    assert _complete(server)["content"] == _complete(server, banned_strings=[])["content"]


def test_a_ban_leaves_the_next_request_unchanged(server):
    # a restored checkpoint must not leak into the slot's next request
    ref = _complete(server)["content"]
    _complete(server, banned_strings=[_a_word(ref)])
    assert _complete(server)["content"] == ref


def test_ban_with_prompt_cache(server):
    # cache_prompt=True reuses the slot state; the ban must still rewind correctly on top of it
    ref = _complete(server, cache_prompt=True)["content"]
    word = _a_word(ref)
    at = ref.lower().find(word.lower())
    got = _complete(server, cache_prompt=True, banned_strings=[word])["content"]
    assert word.lower() not in got.lower(), (word, got)
    assert got[:at] == ref[:at], (at, ref, got)
