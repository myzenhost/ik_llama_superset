"""Per-client request rate limit --rate-limit CALLS/PERIOD / --rate-limit-key (row
S69-MLFLOW-PICK-PER-CLIENT-RATE-LIMIT-2026-10-15-01), and Retry-After on the 503 admission answer (row
S69-LOCALAI-PICK-RETRY-AFTER-HEADERS-2026-09-29-01).

POST /tokenize is the counted request: it is a real POST route but does no generation, so a window of a few seconds
holds many requests. On a build without the feature the server refuses to start with an unknown flag, so every
limited test fails there; the flag-off test is the control that passes on both builds.
"""
import time

import pytest
from utils import ServerPreset, parallel_function_calls

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.server_port = 18436  # NOT 8081 (live serving) and not 8088/8090 (claimed by other jobs)


def _tok(headers=None):
    return server.make_request("POST", "/tokenize", data={"content": "hello"}, headers=headers)


def _retry_after(res):
    v = res.headers.get("Retry-After")
    assert v is not None, res.headers
    assert v.isdigit(), v
    return int(v)


def test_flag_off_no_limit():
    server.start()
    codes = [_tok().status_code for _ in range(8)]
    assert codes == [200] * 8, codes
    assert "X-RateLimit-Limit" not in _tok().headers


def test_n_plus_one_gets_429_with_retry_after():
    server.extra_args = ["--rate-limit", "2/4s"]
    server.start()
    r1, r2, r3 = _tok(), _tok(), _tok()
    assert [r1.status_code, r2.status_code] == [200, 200]
    assert r1.headers["X-RateLimit-Limit"] == "2"
    assert r1.headers["X-RateLimit-Remaining"] == "1"
    assert r2.headers["X-RateLimit-Remaining"] == "0"
    assert r3.status_code == 429, r3.body
    assert r3.body["error"]["type"] == "rate_limit_error"
    assert r3.body["error"]["code"] == 429
    wait = _retry_after(r3)
    assert 1 <= wait <= 4, wait               # the time left in a 4 s window, never 0 and never more than the window
    # reads are not counted: polling /health, /props, /models still answers while the client is over its budget
    assert server.make_request("GET", "/health").status_code == 200
    assert server.make_request("GET", "/props").status_code == 200
    assert server.make_request("GET", "/v1/models").status_code == 200
    # the window ends: served again, a fresh budget
    time.sleep(wait + 0.2)
    r4 = _tok()
    assert r4.status_code == 200, r4.body
    assert r4.headers["X-RateLimit-Remaining"] == "1"


def test_other_client_key_unaffected():
    server.api_key = "k1,k2"
    server.extra_args = ["--rate-limit", "2/30s", "--rate-limit-key", "key"]
    server.start()
    a = {"Authorization": "Bearer k1"}
    b = {"X-Api-Key": "k2"}
    assert [_tok(a).status_code for _ in range(3)] == [200, 200, 429]
    # a second client (another key, same address) has its own budget
    assert [_tok(b).status_code for _ in range(2)] == [200, 200]
    assert _tok(b).status_code == 429
    # a made-up key is answered 401 by the key check, before it could open a counter
    assert _tok({"Authorization": "Bearer nope"}).status_code == 401


def test_addr_mode_counts_all_keys_from_one_address_together():
    # control for the test above: keyed by address, two keys from 127.0.0.1 share one budget
    server.api_key = "k1,k2"
    server.extra_args = ["--rate-limit", "2/30s"]
    server.start()
    assert _tok({"Authorization": "Bearer k1"}).status_code == 200
    assert _tok({"Authorization": "Bearer k2"}).status_code == 200
    assert _tok({"Authorization": "Bearer k1"}).status_code == 429
    assert _tok({"Authorization": "Bearer k2"}).status_code == 429


def test_admission_503_carries_retry_after():
    server.extra_args = ["--max-queued-requests", "1"]
    server.start()
    data = {"prompt": "Once upon a time", "n_predict": 256, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}
    res = parallel_function_calls([(server.make_request, ("POST", "/completion", data)) for _ in range(6)])
    codes = [r.status_code for r in res]
    assert 200 in codes and 503 in codes, codes
    for r in res:
        if r.status_code == 503:
            assert 1 <= _retry_after(r) <= 60, r.headers
        else:
            assert "Retry-After" not in r.headers     # a 200 carries none


@pytest.mark.parametrize("bad", ["0/1s", "2", "2/fortnight", "x/1s"])
def test_bad_flag_values_refuse_to_start(bad):
    bad_srv = ServerPreset.tinyllama2()
    bad_srv.server_port = 18437
    bad_srv.extra_args = ["--rate-limit", bad]
    with pytest.raises(Exception):
        bad_srv.start(timeout_seconds=15)


def test_bad_key_mode_refuses_to_start():
    # positive control first: a valid mode starts
    server.extra_args = ["--rate-limit", "5/1s", "--rate-limit-key", "addr"]
    server.start()
    server.stop()
    bad_srv = ServerPreset.tinyllama2()
    bad_srv.server_port = 18437
    bad_srv.extra_args = ["--rate-limit", "5/1s", "--rate-limit-key", "header"]
    with pytest.raises(Exception):
        bad_srv.start(timeout_seconds=15)
