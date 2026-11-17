import re
import pytest
from utils import ServerPreset
from utils import use_local_stories260k  # explicit, so static checkers see where it comes from

# Grammar jump-forward (row S69-MLC-LLM-PICK-GRAMMAR-JUMP-FORWARD-2026-10-15-01; algorithm in tools/server/server-jump-forward.h,
# idea from MLC-LLM batch_jump_forward.cc, Apache-2.0, no code copied).
#
# When a user GBNF grammar leaves no choice about the next text, the server appends the tokens spelling it and decodes them in ONE
# batch. The reference for every request is the SAME request with the feature off ("grammar_jump_forward": 0) on the same server:
#   * the text of a fully forced grammar is identical, with fewer llama_decode() calls (llamacpp:n_decode_total) and a
#     llamacpp:grammar_jump_forward_tokens_total that moved,
#   * a request that is not eligible (n_probs, no grammar) does not move the counter and returns the same text.

LITERAL = "the little dog was happy and the little cat was sad"
FORCED = 'root ::= "%s"' % LITERAL
MIXED = 'root ::= "the little " [a-z] [a-z]? [a-z]? " was sad and the little dog was happy"'

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_metrics = True
    use_local_stories260k(server)


def _metric(name: str) -> float:
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(name + " "):
            return float(line.split(" ", 1)[1])
    raise AssertionError(f"{name} not in /metrics")


def _complete(grammar=None, jf=None, **extra):
    data = {"n_predict": 128, "prompt": "Once upon a time", "temperature": 0.0, "cache_prompt": False}
    if grammar is not None:
        data["grammar"] = grammar
    if jf is not None:
        data["grammar_jump_forward"] = jf
    data.update(extra)
    d0 = _metric("llamacpp:n_decode_total")
    j0 = _metric("llamacpp:grammar_jump_forward_tokens_total")
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200
    return res, _metric("llamacpp:n_decode_total") - d0, _metric("llamacpp:grammar_jump_forward_tokens_total") - j0


def test_forced_grammar_same_text_fewer_decodes():
    server.start()
    off, d_off, j_off = _complete(FORCED, jf=0)
    on, d_on, j_on = _complete(FORCED, jf=32)
    assert off.body["content"] == LITERAL                      # the reference run
    assert on.body["content"] == off.body["content"]           # identical text
    print("JUMP-FORWARD forced: decode calls off=%d on=%d, jumped tokens=%d, tokens_predicted off=%s on=%s" % (d_off, d_on, j_on, off.body["tokens_predicted"], on.body["tokens_predicted"]))
    assert j_off == 0                                           # control: off moves nothing
    assert j_on > 0                                             # the jump fired
    assert d_on < d_off, (d_on, d_off)                          # and cost fewer llama_decode() calls
    assert on.body["generation_settings"]["grammar_jump_forward"] == 32
    assert off.body["generation_settings"]["grammar_jump_forward"] == 0


def test_cap_limits_tokens_per_step():
    server.start()
    _, d_big, j_big = _complete(FORCED, jf=64)
    one, d_one, j_one = _complete(FORCED, jf=1)
    assert one.body["content"] == LITERAL
    assert d_big <= d_one                                       # a bigger cap never costs more decodes
    assert j_one > 0 and j_big >= j_one


def test_mixed_grammar_stays_inside_the_grammar():
    server.start()
    on, d_on, j_on = _complete(MIXED, jf=32)
    off, d_off, j_off = _complete(MIXED, jf=0)
    pat = re.compile(r"^the little [a-z]{1,3} was sad and the little dog was happy$")
    assert pat.match(off.body["content"]), off.body["content"]
    assert pat.match(on.body["content"]), on.body["content"]    # the text may differ in the free part, never leaves the grammar
    print("JUMP-FORWARD mixed: decode calls off=%d on=%d, jumped tokens=%d, text off=%r on=%r" % (d_off, d_on, j_on, off.body["content"], on.body["content"]))
    assert j_off == 0 and j_on > 0
    assert d_on <= d_off


def test_alternation_forces_only_the_common_prefix():
    server.start()
    g = 'root ::= "the little " ("dog" | "dot") " was sad"'
    on, _, j_on = _complete(g, jf=32)
    off, _, _ = _complete(g, jf=0)
    assert on.body["content"] in ("the little dog was sad", "the little dot was sad"), on.body["content"]
    assert off.body["content"] in ("the little dog was sad", "the little dot was sad"), off.body["content"]
    assert j_on > 0                                             # "the little do" is forced, the third letter is the model's choice


def test_ineligible_requests_do_not_jump():
    server.start()
    # logprobs asked for (MLC skips these too): same text, counter does not move
    off, _, _ = _complete(FORCED, jf=0, n_probs=2)
    lp, _, j_lp = _complete(FORCED, jf=32, n_probs=2)
    assert j_lp == 0
    assert lp.body["content"] == off.body["content"] == LITERAL
    # no grammar at all: nothing to force
    a, _, j_a = _complete(None, jf=32, n_predict=24)
    b, _, _ = _complete(None, jf=0, n_predict=24)
    assert j_a == 0
    assert a.body["content"] == b.body["content"]


def test_server_flag_is_the_default_and_a_request_can_turn_it_off():
    server.extra_args = ["--grammar-jump-forward", "32"]
    server.start()
    on, d_on, j_on = _complete(FORCED)                          # no request field: the flag applies
    off, d_off, j_off = _complete(FORCED, jf=0)                 # the request wins
    assert on.body["content"] == off.body["content"] == LITERAL
    assert j_on > 0 and j_off == 0
    assert d_on < d_off
    assert on.body["generation_settings"]["grammar_jump_forward"] == -1   # the request value as sent / defaulted


def test_default_is_off():
    server.start()
    res, _, j = _complete(FORCED)
    assert res.body["content"] == LITERAL
    assert j == 0
    assert res.body["generation_settings"]["grammar_jump_forward"] == -1


def test_streaming_text_is_the_same():
    server.start()
    chunks = []
    for jf in (0, 32):
        text = ""
        for data in server.make_stream_request("POST", "/completion", data={
                "n_predict": 128, "prompt": "Once upon a time", "temperature": 0.0, "grammar": FORCED,
                "grammar_jump_forward": jf, "stream": True, "cache_prompt": False}):
            text += data.get("content", "")
        chunks.append(text)
    assert chunks[0] == chunks[1] == LITERAL
