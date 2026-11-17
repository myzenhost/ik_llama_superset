import pytest
from utils import *

# item 2: --reasoning-tokens, ported from the pre-merge fork's examples/server (row
# S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, item 2).
# SOURCE-READ: examples/server/server-context.cpp:353-358,1003-1090 @ 35f0cac75,
#              examples/server/server-task.cpp:1153-1200 @ 35f0cac75,
#              examples/server/server-common.cpp:1512-1596 @ 35f0cac75 (get_tokens_exclude_think).
#
# Before this port --reasoning-tokens was not registered in common/arg.cpp at all, so passing it
# made llama-server refuse to start (ServerProcess.start() raises RuntimeError when the child process
# exits immediately). That is the fail-before control for the flag-acceptance tests.

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0


def test_reasoning_tokens_none_flag_accepted():
    global server
    server.extra_args = ["--reasoning-tokens", "none"]
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_reasoning_tokens_auto_flag_accepted():
    global server
    server.extra_args = ["--reasoning-tokens", "auto"]
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_reasoning_tokens_custom_delimiters_flag_accepted():
    global server
    server.extra_args = ["--reasoning-tokens", "<r>,</r>"]
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
    })
    assert res.status_code == 200


def _seed_and_query(reasoning_tokens_mode: str) -> int:
    """Seed slot 0 and slot 1 with prompts sharing a <think> body but different visible text, then
    send an auto-selected (no id_slot) query whose <think> body matches slot 0's exactly, but whose
    VISIBLE text matches slot 1's far more closely. Returns which slot handled the query.

    With raw token comparison (--reasoning-tokens none), the identical <think> body makes slot 0's
    prompt share a long literal prefix with the query (they only diverge once the visible tail
    starts), so slot 0 should win. With reasoning tokens excluded (the default, "auto"), the <think>
    body is stripped before comparing, so the match is decided by the visible text alone -- which
    matches slot 1 far more closely -- so slot 1 should win instead.
    """
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    server.n_predict = 8
    # a near-zero similarity threshold makes the LCP-similarity slot-selection path always decide
    # (over the LRU fallback) as long as either candidate slot has ANY nonzero prefix match, so the
    # test isn't sensitive to exactly how many tokens the tiny test model's vocab happens to share.
    server.extra_args = ["--reasoning-tokens", reasoning_tokens_mode, "-sps", "0.01"]
    server.start()

    visible_tail_1 = " one two three four five six seven eight nine ten eleven twelve thirteen fourteen"

    res0 = server.make_request("POST", "/completion", data={
        "prompt": "<think>alpha alpha alpha alpha alpha</think> zero AAAAAAA tail",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res0.status_code == 200

    res1 = server.make_request("POST", "/completion", data={
        "prompt": "<think>completely unrelated reasoning here</think> BBBBBBB" + visible_tail_1,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res1.status_code == 200

    query = server.make_request("POST", "/completion", data={
        "prompt": "<think>alpha alpha alpha alpha alpha</think> BBBBBBB" + visible_tail_1,
        "cache_prompt": True,
    })
    assert query.status_code == 200
    id_slot = query.body["id_slot"]

    # This helper is called TWICE per test (once per --reasoning-tokens mode), each time starting a
    # fresh server on the SAME port. conftest.py's stop_server_after_each_test only stops servers
    # between TEST FUNCTIONS, not between calls within one -- so without an explicit stop() here, the
    # first server keeps listening on the port while the second one starts, and the second call's own
    # query can silently land on the FIRST (still-alive, wrong-mode) server instead of its own,
    # making both calls appear to return the same slot regardless of --reasoning-tokens. Measured: a
    # raw urllib diagnostic that fully tears down each server between runs saw "none" -> slot 0 and
    # "auto" -> slot 1 correctly, while this same in-process pytest scenario returned slot 0 twice
    # until this stop() was added.
    server.stop()
    return id_slot


def test_reasoning_tokens_changes_slot_selection():
    slot_raw     = _seed_and_query("none")
    slot_exclude = _seed_and_query("auto")
    print(f"slot_raw={slot_raw} slot_exclude={slot_exclude}")

    # explicit expected slots (not just inequality), so a failure here says WHICH arm is wrong:
    # raw comparison must favor slot 0 (matching <think> body, identical literal prefix up to the
    # visible tail), excluding reasoning tokens must favor slot 1 (matching VISIBLE text instead).
    assert slot_raw == 0, f"raw-token comparison should favor slot 0 (matching <think> body), got {slot_raw}"
    assert slot_exclude == 1, f"reasoning-token exclusion should favor slot 1 (matching visible text), got {slot_exclude}"
