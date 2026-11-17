import threading
import time

import pytest
from utils import ServerPreset, use_local_stories260k

# Per-request queue deadline (row S69-Q9-PICK-REQUEST-QUEUE-DEADLINE-2026-09-30-01).
# One slot is held by a long generation. A second request that finds no slot is deferred. With `max_queue_wait_ms` set it is answered
# 503 once the wait passes the limit instead of waiting for the slot; without the field (or with 0) it waits and then completes.
# The running request is never touched. The server-wide default is `--max-queue-wait-ms`; an explicit 0 in the request overrides it.

N_LONG = 600
N_SHORT = 8

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_ctx = 4096
    server.n_slots = 1
    server.n_batch = 64
    server.server_slots = True  # the helper below polls GET /slots to know the long request holds the slot


def _long_then_second(second_body):
    """Start a long generation, wait until /slots shows it processing, then send the second request. Returns
    (long_response, second_response, long_was_running_when_second_was_sent, long_was_running_when_second_returned)."""
    box = {}
    done = threading.Event()

    def run_long():
        box["long"] = server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time", "n_predict": N_LONG, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False},
            timeout=300)
        done.set()

    t = threading.Thread(target=run_long)
    t.start()
    deadline = time.time() + 20
    busy = False
    while time.time() < deadline and not busy:
        slots = server.make_request("GET", "/slots").body
        busy = any(s.get("is_processing") for s in slots)
        time.sleep(0.01)
    assert busy, "control: the long request must be holding the slot before the second one is sent"
    running_at_send = not done.is_set()
    body = {"prompt": "Once upon a time", "n_predict": N_SHORT, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}
    body.update(second_body)
    second = server.make_request("POST", "/completion", data=body, timeout=300)
    running_at_return = not done.is_set()
    t.join()
    return box["long"], second, running_at_send, running_at_return


def test_request_deadline_answers_503_and_leaves_the_running_request_alone():
    global server
    server.start()
    long_res, second, at_send, at_return = _long_then_second({"max_queue_wait_ms": 50})
    assert at_send and at_return, "precondition: the long request must hold the slot while the second is sent and answered (raise N_LONG)"
    assert second.status_code == 503, (second.status_code, second.body)
    assert "max_queue_wait_ms" in second.body["error"]["message"], second.body
    assert long_res.status_code == 200
    assert long_res.body["tokens_predicted"] == N_LONG


def test_control_without_the_field_waits_and_completes():
    global server
    server.start()
    long_res, second, at_send, at_return = _long_then_second({})
    assert at_send, "precondition: the second request must have had to wait for the long one (raise N_LONG)"
    assert second.status_code == 200, (second.status_code, second.body)
    assert second.body["tokens_predicted"] == N_SHORT
    assert long_res.body["tokens_predicted"] == N_LONG
    assert at_return is False, "the second request must have been answered only after the long one released the slot"


def test_control_generous_deadline_still_completes():
    global server
    server.start()
    long_res, second, at_send, _ = _long_then_second({"max_queue_wait_ms": 600000})
    assert at_send
    assert second.status_code == 200, (second.status_code, second.body)
    assert second.body["tokens_predicted"] == N_SHORT


def test_server_default_flag_applies_and_request_zero_overrides_it():
    global server
    server.extra_args = ["--max-queue-wait-ms", "50"]
    server.start()
    _, dropped, at_send, at_return = _long_then_second({})
    assert at_send and at_return, "precondition: the long request must hold the slot (raise N_LONG)"
    assert dropped.status_code == 503, (dropped.status_code, dropped.body)
    server.stop()
    server.start()
    _, waits, at_send, _ = _long_then_second({"max_queue_wait_ms": 0})
    assert at_send
    assert waits.status_code == 200, (waits.status_code, waits.body)
    assert waits.body["tokens_predicted"] == N_SHORT


def test_the_deadline_is_echoed_in_generation_settings():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 2, "max_queue_wait_ms": 1234})
    assert res.status_code == 200
    assert res.body["generation_settings"]["max_queue_wait_ms"] == 1234, res.body["generation_settings"]
