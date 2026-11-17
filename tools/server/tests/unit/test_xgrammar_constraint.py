import json
import os
import re

import pytest
from utils import ServerPreset, use_local_stories260k

# XGrammar ("%xgrammar <kind>\n<body>") as the second sampler-shaped constraint backend next to llguidance
# (row S69-ADD-TGI-ALTERNATIVES-LLGUIDANCE-PAGEDKV-RADIX-SYNTHID-2026-09-27-01; common/xgrammar.cpp).
# Needs a -DLLAMA_XGRAMMAR=ON server binary; set LLAMA_TEST_XGRAMMAR=1 to run the constrained cases.
# test_xgrammar_refused_when_not_built runs on EVERY build: without XGrammar the request must be refused (400),
# never run unconstrained and never abort the server.

XGR = os.environ.get("LLAMA_TEST_XGRAMMAR") == "1"
needs_xgr = pytest.mark.skipif(not XGR, reason="set LLAMA_TEST_XGRAMMAR=1 with a -DLLAMA_XGRAMMAR=ON server binary")

server = ServerPreset.tinyllama2()
PROMPT = "Once upon a time"

CASES = [
    # (grammar request string, python regex every output must fully match)
    ('%xgrammar ebnf\nroot ::= "yes" | "no" | "maybe"', r"(yes|no|maybe)"),
    ('%xgrammar regex\n[0-9]{3}-[0-9]{4}', r"[0-9]{3}-[0-9]{4}"),
    ('%xgrammar lark\nstart: WORD (" " WORD)?\nWORD: /[A-Z][a-z]{2,6}/', r"[A-Z][a-z]{2,6}( [A-Z][a-z]{2,6})?"),
]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)


@needs_xgr
@pytest.mark.parametrize("grammar,pattern", CASES)
def test_xgrammar_output_fully_matches(grammar, pattern):
    global server
    server.start()
    for seed in (1, 2, 3):
        res = server.make_request("POST", "/completion", data={
            "prompt": PROMPT, "n_predict": 32, "grammar": grammar, "temperature": 1.0, "seed": seed})
        assert res.status_code == 200, res.body
        assert re.fullmatch(pattern, res.body["content"]), (grammar, seed, res.body["content"])


@needs_xgr
def test_xgrammar_json_schema_is_valid_json():
    global server
    server.start()
    # boolean, not integer: an integer has no length bound, and stories260K ran "-6000..." to the token budget
    schema = {"type": "object", "properties": {"n": {"type": "boolean"}}, "required": ["n"],
              "additionalProperties": False}
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 48, "temperature": 1.0, "seed": 5,
        "grammar": "%xgrammar json_schema\n" + json.dumps(schema)})
    assert res.status_code == 200, res.body
    out = json.loads(res.body["content"])  # must PARSE, not merely look like JSON
    assert isinstance(out, dict) and set(out) == {"n"} and isinstance(out["n"], bool), out


@needs_xgr
def test_xgrammar_json_has_fixed_separators_by_default_and_any_whitespace_opts_in():
    # Default = XGrammar's fixed separators ", " and ": " (json_schema_converter.cc:1770-1781), i.e. exactly what
    # json.dumps writes. MEASURED before this default: any_whitespace=true let stories260K emit '{ \n"n"\n\n\n ...'
    # until the budget ran out.
    global server
    server.start()
    schema = {"type": "object", "properties": {"n": {"type": "boolean"}}, "required": ["n"],
              "additionalProperties": False}
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 48, "temperature": 1.0, "seed": 5,
        "grammar": "%xgrammar json_schema\n" + json.dumps(schema)})
    assert res.status_code == 200, res.body
    content = res.body["content"]
    assert content == json.dumps(json.loads(content)), content
    # the opt-in is accepted (content may contain whitespace; only the request contract is checked here)
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "temperature": 1.0, "seed": 5,
        "grammar": "%xgrammar json_schema any_whitespace\n" + json.dumps(schema)})
    assert res.status_code == 200, res.body
    # an unknown option is refused, not ignored
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "grammar": "%xgrammar json_schema bogus_option\n{}"})
    assert res.status_code == 400, res.body


@needs_xgr
def test_xgrammar_bad_grammar_is_refused_not_ignored():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "grammar": '%xgrammar ebnf\nroot ::= undefined_rule_xyz'})
    assert res.status_code == 400, res.body
    assert "xgrammar" in res.body["error"]["message"]


@needs_xgr
def test_xgrammar_unknown_kind_is_refused():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "grammar": '%xgrammar nonsense\nroot ::= "a"'})
    assert res.status_code == 400, res.body


@pytest.mark.skipif(XGR, reason="this binary HAS XGrammar; the refusal case is for builds without it")
def test_xgrammar_refused_when_not_built():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "grammar": '%xgrammar ebnf\nroot ::= "yes"'})
    assert res.status_code == 400, res.body
    # and the server is still alive for the next request
    assert server.make_request("GET", "/health").status_code == 200
