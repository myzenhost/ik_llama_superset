# vLLM-engine merge: request validation that used to answer 500 or a silent 200.
#  - malformed JSON body -> 400        (vllm chat_completion/api_router.py:41)
#  - /detokenize bad ids -> 400        (vllm serve/tokenize/protocol.py:164-170, renderers/base.py:449-455)
#  - response_format json_schema without json_schema -> 400 (vllm chat_completion/protocol.py:761-792)
#  - repeat_penalty <= 0 -> 400        (vllm sampling_params.py)
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tiny_random_model import PROMPT, tiny_server  # noqa: E402


@pytest.fixture
def srv(tmp_path):
    server = tiny_server(tmp_path)
    server.start()
    yield server
    server.stop()


def _raw(server, path, body):
    import requests
    return requests.post(f"http://{server.server_host}:{server.server_port}{path}", data=body,
                         headers={"Content-Type": "application/json"}, timeout=30)


@pytest.mark.parametrize("path", ["/completion", "/v1/completions", "/v1/chat/completions", "/tokenize", "/detokenize"])
def test_malformed_json_is_400(srv, path):
    for body in ("{not json", "[1]"):
        r = _raw(srv, path, body)
        assert r.status_code == 400, (path, body, r.status_code, r.text)
        assert r.json()["error"]["type"] == "invalid_request_error"


def test_detokenize_validates_ids(srv):
    ok = srv.make_request("POST", "/detokenize", data={"tokens": [4, 5]})
    assert ok.status_code == 200                                    # control
    for bad in ({"tokens": [-1]}, {"tokens": [999999999]}, {"tokens": "abc"}, {"tokens": [1.5]}, {}):
        r = srv.make_request("POST", "/detokenize", data=bad)
        assert r.status_code == 400, (bad, r.status_code, r.body)


def test_response_format_json_schema_requires_schema_object(srv):
    msgs = [{"role": "user", "content": "x"}]
    for rf in ({"type": "json_schema"}, {"type": "json_schema", "json_schema": "no"}, "text", {"type": "bogus"}):
        r = srv.make_request("POST", "/v1/chat/completions", data={"messages": msgs, "max_tokens": 1, "response_format": rf})
        assert r.status_code == 400, (rf, r.status_code, r.body)


@pytest.mark.parametrize("field", ["repeat_penalty", "repetition_penalty"])
def test_nonpositive_repeat_penalty_is_400(srv, field):
    for v in (0, -1.5):
        r = srv.make_request("POST", "/completion", data={"prompt": PROMPT, "n_predict": 1, field: v})
        assert r.status_code == 400, (field, v, r.status_code, r.body)
    r = srv.make_request("POST", "/completion", data={"prompt": PROMPT, "n_predict": 1, field: 1.1})
    assert r.status_code == 200                                     # control


def test_lora_entries_are_validated(srv):
    base = {"prompt": PROMPT, "n_predict": 1}
    assert srv.make_request("POST", "/completion", data=dict(base, lora=[])).status_code == 200   # control
    for bad in ([{"id": 0, "scale": 1.0}], [1], [{"id": "a", "scale": 1}], {}, [{"scale": 1}]):
        r = srv.make_request("POST", "/completion", data=dict(base, lora=bad))
        assert r.status_code == 400, (bad, r.status_code, r.body)
