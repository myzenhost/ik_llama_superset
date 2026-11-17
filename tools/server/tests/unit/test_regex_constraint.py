import re

import pytest
from utils import *
from utils import use_local_stories260k  # explicit, so static checkers see where it comes from

# Regex-constrained decoding (row S69-SERVER-REGEX-CONSTRAINED-DECODING-2026-09-27-01). Every constrained output
# must FULLY match its regex (re.fullmatch), across several seeds at temperature 1 so the sampler actually explores.
# Negative control: the same prompt unconstrained does not produce matching text, so a pass is the constraint's doing.

server = ServerPreset.tinyllama2()
PROMPT = "Once upon a time"

REGEXES = [
    r"[0-9]{3}-[0-9]{4}",
    r"(yes|no|maybe)",
    r"[A-Z][a-z]{2,6}( [A-Z][a-z]{2,6})?",
    r"\d+\.\d\d",
    r"(ab|cd)+x?",
]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)


@pytest.mark.parametrize("pattern", REGEXES)
def test_regex_output_fully_matches(pattern):
    global server
    server.start()
    for seed in (1, 2, 3):
        res = server.make_request("POST", "/completion", data={
            "prompt": PROMPT, "n_predict": 32, "regex": pattern, "temperature": 1.0, "seed": seed})
        assert res.status_code == 200, res.body
        assert re.fullmatch(pattern, res.body["content"]), (pattern, seed, res.body["content"])


def test_unconstrained_control_does_not_match():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 16, "temperature": 0.0})
    assert res.status_code == 200
    assert not any(re.fullmatch(p, res.body["content"]) for p in REGEXES), res.body["content"]


def test_regex_unsupported_construct_is_refused():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "regex": r"(?=a)abc"})
    assert res.status_code == 400
    assert "regex" in res.body["error"]["message"]


def test_regex_invalid_is_refused():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": PROMPT, "n_predict": 8, "regex": "(ab"})
    assert res.status_code == 400
    assert "regex" in res.body["error"]["message"]


def test_regex_with_grammar_is_refused():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "regex": "a+", "grammar": 'root ::= "a"'})
    assert res.status_code == 400
