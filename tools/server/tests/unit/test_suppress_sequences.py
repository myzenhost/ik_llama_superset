import pytest
from utils import *
from utils import use_local_stories260k  # explicit, so static checkers see where it comes from

# suppress_sequences: CTranslate2's SuppressSequences logits processor (OpenNMT/CTranslate2 @ f2f54b42, MIT, src/decoding_utils.cc:98-150,
# behaviour only): a token sequence of length 1 is banned at every step; a longer one bans its LAST token whenever the history ends with all
# the tokens before it. The expected answers are built here from plain requests with logit_bias (a mechanism the fork already had, which
# shares no code with the new stage): the position where a sequence first applies is found from the greedy baseline, and what the model says
# there with that one token banned is the oracle. The same cases run against the real CTranslate2 engine in tests/test_ct2_sampling_parity.py.

PROMPT = "Once upon a time there was a little girl"
server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 2


def _toks(**extra):
    data = {"prompt": PROMPT, "n_predict": 14, "temperature": 0.0, "top_k": 1, "return_tokens": True, "cache_prompt": False, "ignore_eos": True}
    data.update(extra)
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200, res.body
    return res.body["tokens"]


def test_flag_off_is_bit_identical_and_fires_when_on():
    server.start()
    base = _toks()
    assert len(base) == 14
    assert _toks(suppress_sequences=[]) == base            # field present but empty: nothing changes
    banned = _toks(suppress_sequences=[[base[5]]])          # a single token is banned everywhere
    assert base[5] not in banned, banned                    # FIRED: the token is gone from the whole output
    assert banned[:5] == base[:5] and banned != base
    # the echo of the effective parameters carries the field: the request reached the sampler settings
    res = server.make_request("POST", "/completion", data={"prompt": PROMPT, "n_predict": 1, "suppress_sequences": [[1, 2]]})
    assert res.body["generation_settings"]["suppress_sequences"] == [[1, 2]]


def test_sequence_bans_only_the_last_token_after_the_matching_prefix():
    server.start()
    base = _toks()
    # the sequence (base[3], base[4]) first applies at step 4: history ends with base[3] there
    seq = [base[3], base[4]]
    out = _toks(suppress_sequences=[seq])
    assert out[:4] == base[:4]
    assert out[4] != base[4]
    oracle = _toks(logit_bias=[[base[4], False]], n_predict=5)   # what the model says at step 4 with exactly that token banned
    assert out[4] == oracle[4]
    # CONTROL (a): the prefix must match. A sequence whose first token is not base[3] changes nothing at step 4
    other = (base[3] + 1) % 512
    assert other != base[3]
    assert _toks(suppress_sequences=[[other, base[4]]])[:5] == base[:5]
    # CONTROL (b): only the LAST token is banned, not the prefix: base[3] itself is still generated at step 3
    assert out[3] == base[3]


def test_three_token_sequence_and_a_mix_of_sequences():
    server.start()
    base = _toks()
    seq3 = [base[3], base[4], base[5]]
    out = _toks(suppress_sequences=[seq3])
    assert out[:5] == base[:5] and out[5] != base[5]
    # two sequences at once: the earlier one decides the first divergence
    out2 = _toks(suppress_sequences=[seq3, [base[9]]])
    assert out2[:5] == base[:5] and out2[5] != base[5]
    # a prefix of the history that matches only part of the sequence does nothing: (base[2], base[4], base[5]) never matches
    assert _toks(suppress_sequences=[[base[2], base[4], base[5]]]) == base


def test_string_form_is_tokenized_and_bad_input_is_refused():
    server.start()
    base = _toks()
    text = server.make_request("POST", "/detokenize", data={"tokens": [base[2], base[3]]}).body["content"]
    ids = server.make_request("POST", "/tokenize", data={"content": text, "add_special": False}).body["tokens"]
    out = _toks(suppress_sequences=[text])
    ref = _toks(suppress_sequences=[ids])
    assert out == ref and len(ids) >= 1
    for bad in ("x", [[1.5]], [[100000]], [7]):
        res = server.make_request("POST", "/completion", data={"prompt": PROMPT, "n_predict": 2, "suppress_sequences": bad})
        assert res.status_code == 400, (bad, res.body)


def test_cli_flag_reaches_the_sampler():
    global server
    server.start()
    base = _toks()
    server.stop()
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 2
    server.extra_args = ["--suppress-sequence", "%d,%d" % (base[3], base[4])]
    server.start()
    out = _toks()
    assert out[:4] == base[:4] and out[4] != base[4]    # the flag is the default for every request
    # a request can override it with its own list
    assert _toks(suppress_sequences=[]) == base
