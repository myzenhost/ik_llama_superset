import pytest
from utils import ServerPreset, use_local_stories260k

# koboldcpp tool_call_fix port. SOURCE-READ: engines/koboldcpp/gpttype_adapter.cpp:159 (the id list, "temp ban these id for the first
# 3 tokens generated, to prevent empty replies"), :5783-5794 (filled with every token whose text contains ']'), :7023-7030 (applied
# only while fewer than 3 tokens are out AND the output holds at most one '['). AGPL-3.0: behaviour only, no code copied.
#
# The first port banned those tokens for the WHOLE generation (as logit_bias entries), so a JSON array could never be closed;
# the old third test asserted exactly that. Row S69-KOBOLDCPP-PICK-TOOL-CALL-FIX-FIRST-3-TOKENS-ONLY-2026-09-29-01 corrects it:
# the tokens are masked per step for the first 3 generated tokens only, and never enter logit_bias.

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_port = 8088
    # ROOT-CAUSE: this CPU build's cpp-httplib has no TLS backend -- CMake configure logged
    # "Could NOT find OpenSSL" and this build was not given -DLLAMA_BUILD_BORINGSSL/-LIBRESSL, so
    # llama-server's own HTTPS client is absent, not merely misconfigured.
    # CONFIGS-TRIED: (1) --hf-repo as-is -> server exits "get_repo_commit: error: HTTPS is not
    # supported"; (2) local --model file (this fixture) -> starts fine, same gguf bytes.
    # RESEARCH-DONE: server-side fix is -DLLAMA_OPENSSL=ON + system OpenSSL dev headers, a build-wide
    # toolchain change out of scope for this one test; the test harness's own Python download (which
    # has its own SSL via the interpreter, unrelated to this binary) already fetched the file once.
    # The file is resolved from the model cache (tests/.test_models is gitignored, so a hardcoded path there
    # failed in every fresh checkout); skips with the reason if the model is not cached.
    use_local_stories260k(server)


def _gen(prompt, n_predict, **extra):
    data = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "seed": 42, "cache_prompt": False}
    data.update(extra)
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200
    return res


def test_tool_call_fix_does_not_touch_logit_bias():
    """The ids are masked per step, not parked in logit_bias for the whole request."""
    global server
    server.start()
    res = _gen("Once upon a time", 4, tool_call_fix=True)
    assert res.body["generation_settings"]["logit_bias"] == []
    assert res.body["generation_settings"]["tool_call_fix"] is True


def test_tool_call_fix_false_no_logit_bias():
    global server
    server.start()
    res = _gen("Once upon a time", 4, tool_call_fix=False)
    assert res.body["generation_settings"]["logit_bias"] == []
    assert res.body["generation_settings"]["tool_call_fix"] is False


def _bracket_token_id():
    """A vocab token whose text is ']' , found through the server so the test does not hardcode the tokenizer."""
    for text in ("]", "a]", "1]"):
        res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": False})
        assert res.status_code == 200
        for tok in res.body["tokens"]:
            det = server.make_request("POST", "/detokenize", data={"tokens": [tok]})
            if det.body["content"].strip() == "]":
                return tok
    raise AssertionError("control: the test vocabulary has no token that is exactly ']'")


def test_tool_call_fix_blocks_bracket_in_first_three_tokens():
    """With ']' made the model's clear first choice, the flag keeps it out of the first 3 tokens."""
    global server
    server.start()
    bracket = _bracket_token_id()
    bias = [[bracket, 50.0]]
    base = _gen("Once upon a time", 3, logit_bias=bias, return_tokens=True)
    assert base.body["tokens"][0] == bracket, "control: without the flag the boosted ']' is the first token"
    fixed = _gen("Once upon a time", 3, logit_bias=bias, tool_call_fix=True, return_tokens=True)
    assert bracket not in fixed.body["tokens"], "flag on: no ']' among the first 3 tokens"
    assert fixed.body["tokens"] != base.body["tokens"], "the flag is what changed the opening"


def test_tool_call_fix_lets_bracket_through_after_three_tokens():
    """The old behaviour banned ']' for the whole generation, so an array could never be closed."""
    global server
    server.start()
    bracket = _bracket_token_id()
    bias = [[bracket, 50.0]]
    fixed = _gen("Once upon a time", 8, logit_bias=bias, tool_call_fix=True, return_tokens=True)
    toks = fixed.body["tokens"]
    assert bracket not in toks[:3], "the first 3 tokens are protected"
    assert bracket in toks[3:], "from the fourth token on the boosted ']' is free again"
    # control: without the flag the boosted ']' leads from the very first token
    plain = _gen("Once upon a time", 8, logit_bias=bias, return_tokens=True)
    assert plain.body["tokens"][0] == bracket


