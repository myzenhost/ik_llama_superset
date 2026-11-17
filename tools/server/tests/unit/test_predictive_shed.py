import threading
import time

import pytest
from utils import ServerPreset, use_local_stories260k

# Predictive queue shedding (row S69-BENTOML-PICK-PREDICTIVE-QUEUE-SHEDDING-2026-10-15-01).
# --shed-max-latency-ms N fits service time = a*tokens + b over the last 50 finished requests (the first 2 skipped, trusted from the 5th
# kept) and, while MORE THAN ONE request is queued, answers the OLDEST queued request 503 as soon as its wait plus the predicted service
# time of its prompt + n_predict tokens reaches N -- before the wait alone would. A lone queued request is dropped only when its wait
# alone reaches N (BentoML's rule, dispatcher.py:303-311). Before the fit is trusted only the wait counts.
# Timing on this CPU (measured 2026-10-16 with the baseline build under the light lane): decode 4.3-23 ms/token depending on host load
# (it moved 5x inside one test run), prefill 0.12-0.19 ms/token (0.9-1.2 ms/token on a 5-token cold prompt). The assertions below
# therefore do not depend on the speed: one queued request is sized so large that its predicted service alone exceeds the budget at ANY
# of those speeds (2000 tokens x >= 4 ms = 8 s > 1.5 s), the other so small that it cannot (2 tokens x <= 23 ms).

MAX_MS = 1500
N_HOLD = 800          # 3.2 s at the fastest speed seen, 18 s at the slowest: holds the only slot while two requests queue behind it
N_HUGE = 2000         # predicted service >= 8 s
N_TINY = 2            # predicted service <= 0.2 s
WARM = [4, 4, 10, 20, 30, 40, 50]   # 7 finished requests: 2 skipped + 5 kept = the fit is trusted on the last one

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_ctx = 8192
    server.n_slots = 1
    server.n_batch = 64
    server.server_slots = True
    server.server_port = 18434  # NOT 8081: that port is the live serving instance on this host
    server.extra_args = ["--shed-max-latency-ms", str(MAX_MS)]


def _body(n):
    return {"prompt": "Once upon a time", "n_predict": n, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}


def _warm_up():
    for n in WARM:
        assert server.make_request("POST", "/completion", data=_body(n), timeout=120).status_code == 200


def _hold_and_queue_two(n_first, n_second, n_hold=N_HOLD):
    """A holds the slot; B1 then B2 queue behind it. Returns {name: (response, seconds from its own send until answered)}."""
    out = {}

    def run(name, n):
        t0 = time.time()
        r = server.make_request("POST", "/completion", data=_body(n), timeout=600)
        out[name] = (r, time.time() - t0)

    ta = threading.Thread(target=run, args=("A", n_hold))
    ta.start()
    end = time.time() + 20
    while time.time() < end and not any(s.get("is_processing") for s in server.make_request("GET", "/slots").body):
        time.sleep(0.005)
    t1 = threading.Thread(target=run, args=("B1", n_first))
    t1.start()
    time.sleep(0.05)
    t2 = threading.Thread(target=run, args=("B2", n_second))
    t2.start()
    for t in (t1, t2, ta):
        t.join()
    return out


def test_a_queued_request_predicted_to_miss_the_budget_is_shed_at_once():
    server.start()
    _warm_up()
    out = _hold_and_queue_two(N_HUGE, N_TINY)
    b1, t_b1 = out["B1"]
    assert out["A"][1] > MAX_MS / 1000.0 + 0.5, ("precondition: A must hold the slot past the budget", out["A"][1])
    assert b1.status_code == 503 and "shed" in b1.body["error"]["message"], (b1.status_code, str(b1.body)[:200])
    # the wait alone (a fraction of a second) is far below the 1.5 s budget: only the prediction can have dropped it
    assert t_b1 < MAX_MS / 1000.0 - 0.5, t_b1


def test_a_queued_request_predicted_to_fit_waits_until_the_budget_is_nearly_spent():
    server.start()
    _warm_up()
    out = _hold_and_queue_two(N_TINY, N_TINY)
    b1, t_b1 = out["B1"]
    b2, t_b2 = out["B2"]
    assert out["A"][1] > MAX_MS / 1000.0 + 0.5, ("precondition: A must hold the slot past the budget", out["A"][1])
    assert out["A"][0].status_code == 200 and out["A"][0].body["tokens_predicted"] == N_HOLD
    assert b1.status_code == 503 and "shed" in b1.body["error"]["message"], (b1.status_code, str(b1.body)[:200])
    # wait + a tiny predicted service reaches 1.5 s only just before the wait alone does: not early
    assert t_b1 >= MAX_MS / 1000.0 - 0.45, t_b1
    # the lone survivor is dropped only by its wait alone
    assert b2.status_code == 503 and "shed" in b2.body["error"]["message"], (b2.status_code, str(b2.body)[:200])
    assert t_b2 >= MAX_MS / 1000.0 - 0.05, t_b2


def test_before_the_fit_is_trusted_the_prediction_is_not_used():
    server.start()  # cold: no finished request yet, so even a huge request is judged by its wait alone
    out = _hold_and_queue_two(N_HUGE, N_TINY)
    b1, t_b1 = out["B1"]
    assert out["A"][1] > MAX_MS / 1000.0 + 0.5, ("precondition: A must hold the slot past the budget", out["A"][1])
    assert b1.status_code == 503, (b1.status_code, str(b1.body)[:200])
    assert t_b1 >= MAX_MS / 1000.0 - 0.05, ("a cold fit must not shed early", t_b1)


def test_generous_budget_sheds_nobody():
    global server
    server.extra_args = ["--shed-max-latency-ms", "600000"]
    server.start()
    _warm_up()
    out = _hold_and_queue_two(8, 8, n_hold=60)
    assert [out[k][0].status_code for k in ("A", "B1", "B2")] == [200, 200, 200]
    assert out["B1"][0].body["tokens_predicted"] == 8 and out["B2"][0].body["tokens_predicted"] == 8
