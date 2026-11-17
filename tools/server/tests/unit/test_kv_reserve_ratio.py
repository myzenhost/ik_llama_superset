import threading
import time

import pytest
from utils import ServerPreset, use_local_stories260k

# KV reserve ratio (row S69-RTP-LLM-PICK-KV-RESERVE-RATIO-2026-10-15-01).
# --kv-reserve-ratio P with --kv-unified: a NEW request is admitted only if, after its prompt, at least P percent of the shared pool
# stays free for the requests already running; otherwise it waits for a slot release. On an idle server the request is always
# admitted (rtp-llm answers "permanent resource exhausted" for one that can never fit beside the reserve; here that would starve).
# Two slots, a 512-cell pool: A (prompt ~5 tokens, 150 generated) holds up to ~155 cells. B is tiny (5 + 5 tokens, ~0.1 s).
#   ratio 0   -> B runs beside A and finishes long before A;
#   ratio 99  -> reserve = 506 cells, free < reserve while A runs, so B waits for A to finish.

N_A = 150   # 0.6-3 s at the 4-20 ms/token this CPU shows for stories260K depending on host load
N_B = 5

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_ctx = 512
    server.n_slots = 2
    server.n_batch = 64
    server.kv_unified = True
    server.server_slots = True
    server.server_port = 18434  # NOT 8081: that port is the live serving instance on this host


def _a_then_b():
    """Returns (A finished before B returned?, B response, A response)."""
    box = {}
    t_end = {}

    def run_a():
        box["A"] = server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time", "n_predict": N_A, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}, timeout=300)
        t_end["A"] = time.time()

    t = threading.Thread(target=run_a)
    t.start()
    end = time.time() + 20
    while time.time() < end and not any(s.get("is_processing") for s in server.make_request("GET", "/slots").body):
        time.sleep(0.005)
    b = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": N_B, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}, timeout=300)
    t_end["B"] = time.time()
    t.join()
    return t_end["A"] <= t_end["B"], b, box["A"]


def test_control_without_a_reserve_the_small_request_runs_beside_the_big_one():
    global server
    server.start()
    a_first, b, a = _a_then_b()
    assert b.status_code == 200 and b.body["tokens_predicted"] == N_B
    assert a.status_code == 200 and a.body["tokens_predicted"] == N_A
    assert not a_first, "without a reserve B must finish while A is still generating"


def test_a_modest_reserve_still_lets_it_run_beside():
    global server
    server.extra_args = ["--kv-reserve-ratio", "5"]
    server.start()
    a_first, b, a = _a_then_b()
    assert b.status_code == 200 and a.status_code == 200
    assert not a_first, "5 percent of 512 cells (25) is free beside A, so B must not wait"


def test_a_large_reserve_makes_the_new_request_wait_for_the_running_one():
    global server
    server.extra_args = ["--kv-reserve-ratio", "99"]
    server.start()
    a_first, b, a = _a_then_b()
    assert b.status_code == 200 and b.body["tokens_predicted"] == N_B
    assert a.status_code == 200 and a.body["tokens_predicted"] == N_A
    assert a_first, "with 99 percent held back B must wait until A has finished and released its cells"


def test_idle_server_always_admits_even_when_the_reserve_exceeds_the_free_pool():
    global server
    server.extra_args = ["--kv-reserve-ratio", "100"]
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": N_B, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}, timeout=60)
    assert res.status_code == 200 and res.body["tokens_predicted"] == N_B
