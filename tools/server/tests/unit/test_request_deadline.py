import threading
import time

import pytest
from utils import ServerPreset, use_local_stories260k

# One end-to-end deadline per request (row S69-TFSERVING-PICK-END-TO-END-DEADLINE-2026-10-15-01).
# `t_deadline_ms` (server default `--request-deadline-ms`) is ONE budget counted from the arrival of the request and spent by the queue
# wait, the prompt and the generation together:
#   * still queued, or still reading its prompt, when it runs out -> HTTP 503 (the slot, if it had one, is freed);
#   * generating when it runs out -> stopped, finish_reason "deadline" (native `stop_type` "deadline"), the text so far is returned.
# The row's test: deadline 1 s, queue wait 0.6 s -> generation stops at about 0.4 s. One slot is held by a first request for about 0.6 s;
# the second request, sent while it runs, carries the deadline. Without the feature the field is ignored and it runs to its n_predict.
# stories260K decodes at 4-20 ms/token on this CPU depending on host load (measured 2026-10-16), so the token count that holds the slot
# is derived from a calibration request made on the same server instead of being fixed.

N_BIG = 600           # n_predict of the request that must be cut: 2.4-12 s if nothing cuts it (4-20 ms/token)
LONG_PROMPT = " ".join(["Once upon a time there was a little girl"] * 50)  # 551 tokens with the stories260K vocabulary

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_ctx = 8192
    server.n_slots = 1
    server.n_batch = 64
    server.server_slots = True  # the helper below polls GET /slots to know the first request holds the slot
    server.server_port = 18434  # NOT 8081: that port is the live serving instance on this host


def _ms_per_token():
    r = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 40, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False, "t_deadline_ms": 0}, timeout=120)
    assert r.status_code == 200
    t = r.body["timings"]
    return max(1.0, t["predicted_ms"] / max(1, t["predicted_n"]))


def _hold_then(second_body, hold_ms, second_path="/completion"):
    """Calibrate, start a request that holds the only slot for about `hold_ms`, wait until /slots shows it processing, then send the
    second one. Returns (hold_response, second_response, seconds_the_second_took, ms_the_hold_took, hold_still_running_at_return)."""
    n_hold = int(min(3000, max(10, hold_ms / _ms_per_token())))
    box = {}
    done = threading.Event()

    def run_hold():
        t0 = time.time()
        box["hold"] = server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time", "n_predict": n_hold, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False,
            "t_deadline_ms": 0}, timeout=300)
        box["hold_ms"] = (time.time() - t0) * 1000
        done.set()

    t = threading.Thread(target=run_hold)
    t.start()
    end = time.time() + 20
    busy = False
    while time.time() < end and not busy:
        busy = any(s.get("is_processing") for s in server.make_request("GET", "/slots").body)
        time.sleep(0.002)
    assert busy, "control: the first request must be holding the slot before the second one is sent"
    t0 = time.time()
    second = server.make_request("POST", second_path, data=second_body, timeout=300)
    took = time.time() - t0
    running_at_return = not done.is_set()
    t.join()
    return box["hold"], second, took, box["hold_ms"], running_at_return, n_hold


def _body(**kw):
    b = {"prompt": "Once upon a time", "n_predict": N_BIG, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}
    b.update(kw)
    return b


def test_deadline_counts_the_queue_wait_and_stops_the_generation_at_the_budget():
    global server
    server.start()
    # The row says 0.6 s of queue wait; the hold is aimed lower (0.4 s) because the decode speed of this CPU moved 1.7x between the
    # calibration request and the held one in a measured run (hold 1028 ms for 600 aimed), and a hold that eats the whole budget
    # leaves nothing to generate with.
    hold, second, took, hold_ms, _, n_hold = _hold_then(_body(t_deadline_ms=1000), 400)
    assert hold.status_code == 200 and hold.body["tokens_predicted"] == n_hold
    assert hold_ms < 900, ("precondition: the first request must release the slot well inside the budget", hold_ms)
    assert second.status_code == 200, (second.status_code, second.body)
    assert second.body["stop_type"] == "deadline", (second.body["stop_type"], second.body["tokens_predicted"], took)
    assert 0 < second.body["tokens_predicted"] < N_BIG
    # queue wait + generation = ONE second, whatever the split. The budget is checked once per generated token, so it can be overshot
    # by at most the longest gap between two tokens of this very request (timings.itl_max_ms; a loaded host stalls a step for 400+ ms).
    itl_max_s = second.body["timings"]["itl_max_ms"] / 1000.0
    assert 0.9 <= took <= 1.0 + itl_max_s + 0.3, (took, itl_max_s)
    # ...and the split is what the row says: it waited about as long as the first request ran, then generated for the rest
    gen_ms = second.body["timings"]["predicted_ms"]
    overshoot_ms = gen_ms + hold_ms - 1000
    assert -250 <= overshoot_ms <= second.body["timings"]["itl_max_ms"] + 250, ("generation ms", gen_ms, "hold ms", hold_ms, second.body["timings"])


def test_a_request_still_queued_at_its_deadline_is_answered_503_and_the_running_one_is_untouched():
    global server
    server.start()
    hold, second, took, hold_ms, hold_running, n_hold = _hold_then(_body(t_deadline_ms=300, n_predict=8), 2500)
    assert hold_running, "precondition: the first request must still hold the slot when the second is answered"
    assert second.status_code == 503, (second.status_code, second.body)
    assert "t_deadline_ms" in second.body["error"]["message"], second.body
    assert took < 1.0, took
    assert hold.status_code == 200 and hold.body["tokens_predicted"] == n_hold


def test_control_without_a_deadline_waits_and_runs_to_its_limit():
    global server
    server.start()
    hold, second, took, hold_ms, hold_running, _ = _hold_then(_body(n_predict=8), 1500)
    assert second.status_code == 200 and second.body["tokens_predicted"] == 8 and second.body["stop_type"] == "limit"
    assert not hold_running, "the second request must have been answered only after the first released the slot"


def test_server_default_budget_and_per_request_override():
    global server
    server.extra_args = ["--request-deadline-ms", "300"]
    server.start()
    # the server default applies to a request that says nothing...
    _, second, _, _, _, _ = _hold_then(_body(n_predict=8), 2000)
    assert second.status_code == 503 and "t_deadline_ms" in second.body["error"]["message"], (second.status_code, second.body)
    # ...and an explicit 0 in the request overrides it (no budget). The hold must itself be a request with a budget of 0.
    r = server.make_request("POST", "/completion", data=_body(n_predict=8, t_deadline_ms=0))
    assert r.status_code == 200 and r.body["tokens_predicted"] == 8, (r.status_code, r.body)


def test_oai_completions_reports_finish_reason_deadline():
    global server
    server.start()
    res = server.make_request("POST", "/v1/completions", data={
        "prompt": "Once upon a time", "max_tokens": N_BIG, "ignore_eos": True, "temperature": 0.0, "t_deadline_ms": 600},
        timeout=60)
    assert res.status_code == 200, (res.status_code, res.body)
    assert res.body["choices"][0]["finish_reason"] == "deadline", res.body["choices"][0]
    assert 0 < res.body["usage"]["completion_tokens"] < N_BIG


def test_deadline_echoed_in_generation_settings():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data=_body(n_predict=4, t_deadline_ms=5000, t_max_prompt_ms=4000))
    assert res.status_code == 200
    assert res.body["generation_settings"]["t_deadline_ms"] == 5000
    assert res.body["generation_settings"]["t_max_prompt_ms"] == 4000


def test_deadline_that_runs_out_during_the_prompt_answers_503():
    global server
    server.n_batch = 16  # 551 prompt tokens in 35 sub-batches: the prompt phase alone outlasts the budget
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT, "n_predict": 4, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False, "t_deadline_ms": 20},
        timeout=60)
    assert res.status_code == 503, (res.status_code, str(res.body)[:200])
    assert "t_deadline_ms" in res.body["error"]["message"], res.body
    # the slot was freed: the same prompt without a budget completes
    ok = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT, "n_predict": 4, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}, timeout=60)
    assert ok.status_code == 200 and ok.body["tokens_predicted"] == 4


def test_t_max_prompt_ms_limits_the_prompt_phase():
    global server
    server.n_batch = 16
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT, "n_predict": 4, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False, "t_max_prompt_ms": 20},
        timeout=60)
    assert res.status_code == 503, (res.status_code, str(res.body)[:200])
    assert "t_max_prompt_ms" in res.body["error"]["message"], res.body
    # a generous limit changes nothing
    ok = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT, "n_predict": 4, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False, "t_max_prompt_ms": 60000},
        timeout=60)
    assert ok.status_code == 200 and ok.body["tokens_predicted"] == 4
