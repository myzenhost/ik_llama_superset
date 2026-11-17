"""Allowlist sampler: --allowlist-unicode-rule / --allowlist-keyword / --allowlist-keyword-delay / --allowlist-pieces.

The fork's pre-merge server (examples/server/server-context.cpp @ 35f0cac75, :1748-1845 and :4675-4692) masked the
vocabulary per request: a token survives only if every codepoint matches some rule (script common/inherited
codepoints defer to the others). Rule sets are STAGED by keywords: once a keyword appears in the generated text the
next set applies. Merge 279b4bfaf dropped the server half and the flags; ported into tools/server 2026-09-27.

The tiny test model has byte-fallback tokens for every ASCII digit, so a digits-only rule always leaves something
to sample, and a single-codepoint rule ('0' only) forces an exact token -- which is what makes the staged test
deterministic.
"""
import re

import pytest

from utils import ServerPreset

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _complete(n_predict=16, **extra):
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": n_predict, "temperature": 0.0, "seed": 42, **extra})
    assert res.status_code == 200, res.body
    return res.body["content"]


def test_control_without_allowlist_generates_letters():
    server.start()
    content = _complete()
    assert re.search(r"[A-Za-z]", content), content


def test_digits_only_rule():
    server.extra_args = ["--allowlist-unicode-rule", "48..57"]
    server.start()
    content = _complete()
    assert content.strip(), "nothing generated"
    assert re.fullmatch(r"[0-9\s]+", content), content


def test_keyword_advances_to_next_rule_set():
    # stage 0 admits only '0', so the first token is a zero; "0" is the keyword, so stage 1 (a..z) applies next
    server.extra_args = ["--allowlist-unicode-rule", "48..48",
                         "--allowlist-keyword", "0",
                         "--allowlist-unicode-rule", "97..122"]
    server.start()
    content = _complete()
    assert re.fullmatch(r"\s*0[\s0]*[a-z\s]*", content), content
    assert re.search(r"[a-z]", content), f"stage 1 never applied: {content!r}"


def test_keyword_delay_holds_the_first_stage():
    # same stages, but the keyword is not looked for until 8 tokens are decoded: the first 8 tokens stay zeros
    server.extra_args = ["--allowlist-unicode-rule", "48..48",
                         "--allowlist-keyword", "0",
                         "--allowlist-unicode-rule", "97..122",
                         "--allowlist-keyword-delay", "8"]
    server.start()
    content = _complete()
    zeros = re.match(r"[\s0]*", content).group(0)
    assert zeros.count("0") >= 8, content


def test_request_keyword_delay_overrides_the_flag():
    server.extra_args = ["--allowlist-unicode-rule", "48..48",
                         "--allowlist-keyword", "0",
                         "--allowlist-unicode-rule", "97..122",
                         "--allowlist-keyword-delay", "8"]
    server.start()
    content = _complete(allowlist_keyword_delay=0)
    zeros = re.match(r"[\s0]*", content).group(0)
    assert zeros.count("0") < 8, content
    assert re.search(r"[a-z]", content), content
