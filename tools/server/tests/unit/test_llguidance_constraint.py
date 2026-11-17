import os
import re

import pytest
from utils import ServerPreset, use_local_stories260k

# llguidance ("%llguidance" Lark grammars) wired into the fork's common_sampler through the sampler-shaped
# constraint slot (row S69-ADD-TGI-ALTERNATIVES-LLGUIDANCE-PAGEDKV-RADIX-SYNTHID-2026-09-27-01).
# Needs a -DLLAMA_LLGUIDANCE=ON server binary; set LLAMA_TEST_LLGUIDANCE=1 to run (skipped otherwise, because a
# default build ABORTS on a %llguidance grammar by design -- common/sampling.cpp).
#
# Before this change an LLGUIDANCE=ON build did not compile at all (#error in common/sampling.cpp), so there is
# no "before" binary to fail against; the negative controls below are what make a pass meaningful:
#  - the same prompt unconstrained does not produce the allowed strings;
#  - a grammar that does not compile is REFUSED (400), never run unconstrained -- which is what
#    llama_sampler_init_llg did before: it returned a sampler with no matcher and apply() was a no-op.

pytestmark = pytest.mark.skipif(os.environ.get("LLAMA_TEST_LLGUIDANCE") != "1",
                                reason="set LLAMA_TEST_LLGUIDANCE=1 with a -DLLAMA_LLGUIDANCE=ON server binary")

server = ServerPreset.tinyllama2()
PROMPT = "Once upon a time"

GRAMMARS = [
    # (lark grammar, python regex every output must fully match)
    ('%llguidance {}\nstart: "yes" | "no" | "maybe"', r"(yes|no|maybe)"),
    ('%llguidance {}\nstart: /[0-9]{3}-[0-9]{4}/', r"[0-9]{3}-[0-9]{4}"),
    ('%llguidance {}\nstart: word (" " word)?\nword: /[A-Z][a-z]{2,6}/', r"[A-Z][a-z]{2,6}( [A-Z][a-z]{2,6})?"),
]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)


@pytest.mark.parametrize("grammar,pattern", GRAMMARS)
def test_llguidance_output_fully_matches(grammar, pattern):
    global server
    server.start()
    for seed in (1, 2, 3):
        res = server.make_request("POST", "/completion", data={
            "prompt": PROMPT, "n_predict": 32, "grammar": grammar, "temperature": 1.0, "seed": seed})
        assert res.status_code == 200, res.body
        assert re.fullmatch(pattern, res.body["content"]), (grammar, seed, res.body["content"])


def test_unconstrained_control_does_not_match():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 16, "temperature": 0.0})
    assert res.status_code == 200
    assert not any(re.fullmatch(p, res.body["content"]) for _, p in GRAMMARS), res.body["content"]


def test_llguidance_bad_grammar_is_refused_not_ignored():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "grammar": '%llguidance {}\nstart: undefined_rule_xyz'})
    assert res.status_code == 400, res.body
    assert "llguidance" in res.body["error"]["message"]


def test_server_survives_bad_grammar():
    # the refusal must not take the slot down: a valid constrained request right after still works
    global server
    server.start()
    server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "grammar": '%llguidance {}\nstart: undefined_rule_xyz'})
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "grammar": GRAMMARS[0][0], "temperature": 1.0, "seed": 4})
    assert res.status_code == 200, res.body
    assert re.fullmatch(GRAMMARS[0][1], res.body["content"]), res.body["content"]
