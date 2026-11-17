# POST /v1/logits: one prefill-only pass that returns raw LM-head logits (row S69-ENGREV-NEXA-V1-LOGITS-PREFILL-ENDPOINT-2026-11-15-01).
#
# SOURCE-READ: engines/nexa-sdk cli/server/handler/logits.go:19-133 (BSD-3-Clause). Behaviour only, no code copied: input_ids, last_only,
# top_n (omitted = 20, 0 = the full vocabulary, < 0 = 400), empty input = 400; a row is the top-N [token id, logit] pairs by descending
# logit, or with top_n 0 the whole vocabulary indexed by token id. This server also accepts "content" (text) in place of input_ids.
# Oracle: the last row's best token must be the token a one-token greedy /completion of the same ids generates.
import pytest

from utils import ServerPreset, use_local_stories260k  # noqa: E402

server = ServerPreset.tinyllama2()

N_VOCAB = 512


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 2
    server.n_batch = 512


def _ids(srv, text="I believe the meaning of life is"):
    res = srv.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    assert res.status_code == 200
    return res.body["tokens"]


def test_top_n_rows_descending():
    global server
    server.start()
    ids = _ids(server)
    res = server.make_request("POST", "/v1/logits", data={"input_ids": ids, "top_n": 5})
    assert res.status_code == 200
    assert res.body["n_rows"] == len(ids) and len(res.body["rows"]) == len(ids)
    assert res.body["vocab_size"] == N_VOCAB and res.body["top_n"] == 5
    for row in res.body["rows"]:
        assert len(row["token_ids"]) == 5 and len(row["logits"]) == 5
        assert row["logits"] == sorted(row["logits"], reverse=True)
        assert len(set(row["token_ids"])) == 5


def test_default_top_n_is_20():
    global server
    server.start()
    res = server.make_request("POST", "/v1/logits", data={"input_ids": _ids(server), "last_only": True})
    assert res.status_code == 200
    assert res.body["top_n"] == 20
    assert len(res.body["rows"][0]["token_ids"]) == 20


def test_full_vocab_rows():
    global server
    server.start()
    ids = _ids(server)
    res = server.make_request("POST", "/v1/logits", data={"input_ids": ids, "top_n": 0})
    assert res.status_code == 200
    assert res.body["n_rows"] == len(ids)
    for row in res.body["rows"]:
        assert "token_ids" not in row
        assert len(row["logits"]) == N_VOCAB


def test_last_only_one_row_and_argmax_is_greedy_token():
    global server
    server.start()
    ids = _ids(server)
    res = server.make_request("POST", "/v1/logits", data={"input_ids": ids, "last_only": True, "top_n": 3})
    assert res.status_code == 200
    assert res.body["n_rows"] == 1 and len(res.body["rows"]) == 1
    comp = server.make_request("POST", "/completion", data={
        "prompt": ids, "n_predict": 1, "temperature": 0.0, "return_tokens": True, "cache_prompt": False})
    assert comp.status_code == 200
    assert res.body["rows"][0]["token_ids"][0] == comp.body["tokens"][0]
    # the full-vocabulary last row has the same argmax
    full = server.make_request("POST", "/v1/logits", data={"input_ids": ids, "last_only": True, "top_n": 0})
    row = full.body["rows"][0]["logits"]
    assert row.index(max(row)) == comp.body["tokens"][0]


def test_content_in_place_of_input_ids():
    global server
    server.start()
    ids = _ids(server)
    a = server.make_request("POST", "/v1/logits", data={"input_ids": ids, "last_only": True, "top_n": 4})
    b = server.make_request("POST", "/v1/logits", data={"content": "I believe the meaning of life is", "last_only": True, "top_n": 4})
    assert a.status_code == 200 and b.status_code == 200
    assert a.body["rows"][0]["token_ids"] == b.body["rows"][0]["token_ids"]


@pytest.mark.parametrize("body", [
    {},
    {"input_ids": []},
    {"content": ""},
    {"input_ids": [1, 2], "top_n": -1},
    {"input_ids": [1, 999999]},
])
def test_invalid_requests_are_400(body):
    global server
    server.start()
    res = server.make_request("POST", "/v1/logits", data=body)
    assert res.status_code == 400
