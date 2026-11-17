"""Banned strings: rewind and resample instead of emitting a banned string or regex match.

ik_llama.cpp's server had this (banned_strings, banned_regex, banned_regex_case_insensitive, banned_n,
banned_bias, rewind_count_max, banbuffer_size, saturate_predict). The merge onto upstream's server kept
the common params and dropped every consumer, so the fields were silently ignored; restored 2026-09-26.

Every test takes the greedy output as its reference and bans something that reference DOES contain,
so a server that ignores the field fails -- which is what the pre-fix binary did.
"""
import re

import pytest
from utils import *

server = ServerPreset.tinyllama2()

PROMPT = "Once upon a time"
N_PREDICT = 48


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _complete(**extra) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": N_PREDICT, "temperature": 0, "cache_prompt": False, **extra,
    })
    assert res.status_code == 200, res.body
    return res.body


def _a_word(text: str) -> str:
    # the first word of 4+ letters that is not at the very start, so there is a prefix to preserve
    words = re.findall(r"[A-Za-z]{4,}", text)
    assert len(words) >= 2, text
    return words[1]


def test_banned_string_is_not_emitted_and_the_prefix_is_kept():
    server.start()
    ref = _complete()["content"]
    word = _a_word(ref)
    at = ref.lower().find(word.lower())

    got = _complete(banned_strings=[word])
    assert word.lower() not in got["content"].lower()
    # the rewind only changes the text from where the match started
    assert got["content"][:at] == ref[:at]
    assert got["generation_settings"]["banned_strings"] == [word.lower()]


def test_banned_string_matches_case_insensitively():
    server.start()
    ref = _complete()["content"]
    word = _a_word(ref)
    got = _complete(banned_strings=[word.upper()])["content"]
    assert word.lower() not in got.lower()


def test_banned_regex():
    server.start()
    ref = _complete()["content"]
    word = _a_word(ref)
    pattern = re.escape(word[:3]) + "[a-z]+"
    assert re.search(pattern, ref)
    got = _complete(banned_regex=[pattern])["content"]
    assert not re.search(pattern, got)


def test_invalid_regex_is_rejected():
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 4, "banned_regex": ["(unclosed"],
    })
    assert res.status_code == 400


def test_no_bans_is_unchanged():
    # control: an empty list must leave output byte-identical to the unbuffered path
    server.start()
    assert _complete()["content"] == _complete(banned_strings=[])["content"]


def test_n_predict_still_counts_every_token_sent():
    server.start()
    ref = _complete()
    word = _a_word(ref["content"])
    got = _complete(banned_strings=[word], saturate_predict=True)
    assert got["tokens_predicted"] <= N_PREDICT
    assert word.lower() not in got["content"].lower()
