import pytest
from utils import *

# BeeLlama-style owner-only slot failure (server-context.cpp): a slot whose memory is out of sync with its prompt used to
# GGML_ABORT the whole server (every client on the port died). It now fails that one request with HTTP 500 and the server
# keeps serving the other slot. The fault is injected with LLAMA_SERVER_TEST_FAULT_POS_MIN_SLOT=<slot id>, which makes the
# slot see pos_min == -1 the FIRST time a request reuses its cached prefix (n_past > 0); the fault is one-shot, so the
# recovery request can legitimately get its prefix back (prompt cache) and must then succeed.
# Row S69-ENGREV-BEELLAMA-OWNER-ONLY-FAILURE-NO-ABORT-2026-10-27-01.

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.temperature = 0.0
    server.extra_env = {"LLAMA_SERVER_TEST_FAULT_POS_MIN_SLOT": "0"}


def test_pos_min_fault_fails_owner_slot_only():
    global server
    server.start()
    prompt = "What is the capital of France?"

    # slot 0 computes the prompt (n_past = 0, the fault does not fire)
    res = server.make_request("POST", "/completion", data={"prompt": prompt, "id_slot": 0, "cache_prompt": True, "n_predict": 4})
    assert res.status_code == 200

    # same prompt on slot 0 again: reuses the prefix (n_past > 0) -> injected pos_min == -1 -> that request fails, no abort
    res = server.make_request("POST", "/completion", data={"prompt": prompt, "id_slot": 0, "cache_prompt": True, "n_predict": 4})
    assert res.status_code == 500
    assert "slot" in str(res.body)

    # the server is still up and slot 1 serves
    res = server.make_request("GET", "/health")
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", data={"prompt": prompt, "id_slot": 1, "cache_prompt": True, "n_predict": 4})
    assert res.status_code == 200

    # slot 0 was reset, not wedged: the next request on it succeeds (it may reuse a prefix again; the fault fired once)
    res = server.make_request("POST", "/completion", data={"prompt": prompt, "id_slot": 0, "cache_prompt": True, "n_predict": 4})
    assert res.status_code == 200
