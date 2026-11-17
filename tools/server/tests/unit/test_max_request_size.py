"""--max-request-size MiB: cap on the HTTP request body, answered 413 with a JSON error (row
S69-ENGINE-MERGE-KOBOLDCPP-2026-09-09-01, NEW-PICK 2 payload cap; koboldcpp --maxrequestsize, AGPL, behaviour only).

POST /tokenize is the request: a real POST route that does no generation. The padding goes into an unknown JSON field,
so the same body is accepted when it is under the cap. On a build without the feature the server refuses to start with
an unknown flag, so the capped tests fail there; test_default_accepts_a_2mib_body is the control that passes on both.
"""
import json

import pytest
import requests
from utils import ServerPreset

server = ServerPreset.tinyllama2()

MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.server_port = 18437  # NOT 8081 (live serving) and not 8088/8090 (claimed by other jobs)


def _body(total_bytes):
    """a valid /tokenize JSON body of about total_bytes bytes"""
    head = {"content": "hello", "pad": ""}
    base = len(json.dumps(head))
    head["pad"] = "x" * max(0, total_bytes - base)
    return json.dumps(head).encode()


def _post(data, chunked=False):
    url = f"http://{server.server_host}:{server.server_port}/tokenize"
    hdrs = {"Content-Type": "application/json"}
    if chunked:
        def gen():
            step = 256 * 1024
            for i in range(0, len(data), step):
                yield data[i:i + step]
        return requests.post(url, data=gen(), headers=hdrs, timeout=60)
    return requests.post(url, data=data, headers=hdrs, timeout=60)


def test_default_accepts_a_2mib_body():
    server.start()
    r = _post(_body(2 * MIB))
    assert r.status_code == 200, r.text
    assert len(r.json()["tokens"]) > 0


def test_cap_refuses_larger_body_with_json_413():
    server.extra_args = ["--max-request-size", "1"]
    server.start()
    r = _post(_body(2 * MIB))
    assert r.status_code == 413, (r.status_code, r.text[:200])
    err = r.json()["error"]
    assert err["code"] == 413
    assert err["type"] == "invalid_request_error"
    assert "1 MiB" in err["message"]
    # under the cap: the same request shape is served
    ok = _post(_body(MIB // 2))
    assert ok.status_code == 200, ok.text
    # the server is still healthy after refusing
    assert server.make_request("GET", "/health").status_code == 200


def test_cap_applies_to_chunked_bodies():
    server.extra_args = ["--max-request-size", "1"]
    server.start()
    r = _post(_body(2 * MIB), chunked=True)
    assert r.status_code == 413, (r.status_code, r.text[:200])
    ok = _post(_body(MIB // 2), chunked=True)
    assert ok.status_code == 200, ok.text


def test_zero_means_no_cap():
    server.extra_args = ["--max-request-size", "0"]
    server.start()
    r = _post(_body(2 * MIB))
    assert r.status_code == 200, r.text


def test_negative_is_refused_at_startup():
    server.extra_args = ["--max-request-size", "-1"]
    with pytest.raises(Exception):
        server.start(timeout_seconds=15)
