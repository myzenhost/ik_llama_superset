"""Admission ceilings --max-queued-requests / --max-queued-tokens (row
S69-aphrodite-PICK-REQUEST-ADMISSION-BACKPRESSURE-2026-09-28-01).

Without a ceiling every request queues, however many are already waiting. With one, a request that would cross it
is answered 503 at once. A burst of BURST concurrent requests against ONE slot, each generating N_PREDICT tokens,
keeps the first admitted requests in flight far longer than the burst takes to arrive, so the later ones meet a full
ceiling. The no-ceiling control must answer every request 200 -- on a build without the feature the ceiling runs
look exactly like it, so each ceiling assertion fails there.

Every run ends with a LEAK check through GET /props: once all responses are back, nothing may still be counted. A
ticket that is never released would refuse every request after it, forever.
"""
import pytest
from utils import ServerPreset, parallel_function_calls

server = ServerPreset.tinyllama2()

BURST = 8
N_PREDICT = 256


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.server_port = 18434  # NOT 8081: that port is the live serving instance on this host


def _one(stream: bool = False):
    data = {"prompt": "Once upon a time", "n_predict": N_PREDICT, "ignore_eos": True, "temperature": 0.0,
            "cache_prompt": False}
    if not stream:
        return server.make_request("POST", "/completion", data=data)
    # a streamed request: 200 + chunks, or the error the stream helper raises on a non-200 answer
    try:
        n = sum(1 for _ in server.make_stream_request("POST", "/completion", data={**data, "stream": True}))
        return (200, n)
    except Exception as e:  # ServerError carries the status
        return (getattr(e, "code", None) or getattr(e, "status_code", None) or str(e), 0)


def _burst():
    return parallel_function_calls([(_one, ()) for _ in range(BURST)])


def _admission():
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    return res.body["admission"]


def _assert_no_leak():
    a = _admission()
    assert a["in_flight_requests"] == 0, a
    assert a["prefill_tokens"] == 0, a
    return a


def test_no_ceiling_queues_everything():
    server.start()
    codes = [r.status_code for r in _burst()]
    assert codes == [200] * BURST, codes
    a = _assert_no_leak()
    assert a["max_queued_requests"] == 0 and a["rejected"] == 0


def test_request_ceiling_rejects_the_overflow_with_503():
    server.extra_args = ["--max-queued-requests", "2"]
    server.start()
    res = _burst()
    codes = [r.status_code for r in res]
    assert 200 in codes and 503 in codes, codes
    assert codes.count(200) >= 2, codes          # the ceiling admits up to 2 while the burst arrives
    for r in res:
        if r.status_code == 503:
            assert "--max-queued-requests" in r.body["error"]["message"], r.body
    a = _assert_no_leak()
    assert a["admitted"] + a["rejected"] == BURST, a
    assert a["rejected"] == codes.count(503), a
    # capacity comes back: after the burst drains, a single request is served again
    assert _one().status_code == 200


def test_token_ceiling_rejects_while_prefill_is_backed_up():
    server.extra_args = ["--max-queued-tokens", "1"]
    server.start()
    # a lone request is admitted whatever its size: nothing else is in prefill
    assert _one().status_code == 200
    res = _burst()
    codes = [r.status_code for r in res]
    assert 200 in codes and 503 in codes, codes
    for r in res:
        if r.status_code == 503:
            assert "--max-queued-tokens" in r.body["error"]["message"], r.body
    _assert_no_leak()


def test_streaming_releases_its_ticket():
    server.extra_args = ["--max-queued-requests", "2", "--max-queued-tokens", "1000"]
    server.start()
    status, n_chunks = _one(stream=True)
    assert status == 200 and n_chunks > 0
    _assert_no_leak()
    streamed = parallel_function_calls([(_one, (True,)) for _ in range(BURST)])
    assert any(s == 200 for s, _ in streamed), streamed
    _assert_no_leak()


def test_flags_reject_negative_values():
    # positive control first: the flag itself must be accepted (0 = no limit). Without it this test passed on a
    # build that did not know the flag at all -- an unknown flag also makes the server exit (MEASURED 2026-09-28).
    server.extra_args = ["--max-queued-requests", "0", "--max-queued-tokens", "0"]
    server.start()
    assert _admission()["max_queued_requests"] == 0
    server.stop()
    bad = ServerPreset.tinyllama2()
    bad.server_port = 18435
    bad.extra_args = ["--max-queued-requests", "-1"]
    with pytest.raises(Exception):
        bad.start(timeout_seconds=15)
