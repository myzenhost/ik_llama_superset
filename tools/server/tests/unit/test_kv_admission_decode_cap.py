import pytest
from utils import ServerPreset, parallel_function_calls, use_local_stories260k

# --kv-admission-decode-tokens N (row S69-ENGREV-TRNRT-ADAPT-KV-RESERVE-ADMISSION-2026-11-13-01).
# --kv-admission projects every request's WHOLE remaining generation; N caps that projection per request (the
# admission rule of RunanywhereAI/trnrt scheduler.py:440-496: prompt plus at most `admission_decode_tokens` of decode).
#
# Load: 4 concurrent ignore_eos generations of 200 tokens against ONE shared 256-cell pool (the load of
# test_kv_unified_preempt.py). They cannot all fit, so what admission lets in decides the outcome:
#   - no cap (control): only what fits is admitted, nobody is preempted, all four finish in full;
#   - cap 16: each request is projected at prompt + 16, all four are admitted at once, the pool runs dry and the
#     --kv-unified preemption answers at least one of them 503 -- the cap was honoured, and the documented cost of a
#     cap smaller than the real generation is visible;
#   - cap 1000 (above every real generation): the projection is the uncapped one, so the result is the control's.

N = 4
N_PREDICT = 200

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_ctx = 256
    server.n_slots = N
    server.n_batch = 64
    server.kv_unified = True


def _run_concurrent():
    body = {"prompt": "Once upon a time", "n_predict": N_PREDICT, "ignore_eos": True, "temperature": 0.0,
            "cache_prompt": False}
    tasks = [(server.make_request, ("POST", "/completion", dict(body))) for _ in range(N)]
    return parallel_function_calls(tasks)


def _all_finish_in_full(results):
    assert all(r.status_code == 200 for r in results), [(r.status_code, r.body) for r in results]
    for r in results:
        assert r.body["tokens_predicted"] == N_PREDICT, r.body


def test_control_no_cap_defers_and_nobody_is_preempted():
    global server
    server.extra_args = ["--kv-admission"]
    server.start()
    _all_finish_in_full(_run_concurrent())


def test_small_cap_admits_all_at_once_so_the_pool_runs_dry():
    global server
    server.extra_args = ["--kv-admission", "--kv-admission-decode-tokens", "16"]
    server.start()
    results = _run_concurrent()
    codes = [r.status_code for r in results]
    assert 500 not in codes, codes
    preempted = [r for r in results if r.status_code == 503]
    # with the uncapped rule this load never preempts (the control above); a 503 here is the cap at work
    assert len(preempted) >= 1, codes
    for r in preempted:
        assert "preempted" in r.body["error"]["message"], r.body
    ok = [r for r in results if r.status_code == 200]
    assert len(ok) >= 1 and len(ok) + len(preempted) == N, codes
    for r in ok:
        assert r.body["tokens_predicted"] == N_PREDICT, r.body
    # the server still serves afterwards
    res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 8})
    assert res.status_code == 200, res.body


def test_cap_above_the_real_generation_changes_nothing():
    global server
    server.extra_args = ["--kv-admission", "--kv-admission-decode-tokens", "1000"]
    server.start()
    _all_finish_in_full(_run_concurrent())


def test_short_requests_with_a_cap_run_without_preemption():
    # the case the cap is for: generations that stay under N. 4 x (prompt + 24 tokens) fits the 256-cell pool
    # whatever the admission order, so every request finishes and nobody is preempted.
    global server
    server.extra_args = ["--kv-admission", "--kv-admission-decode-tokens", "32"]
    server.start()
    body = {"prompt": "Once upon a time", "n_predict": 24, "ignore_eos": True, "temperature": 0.0,
            "cache_prompt": False}
    results = parallel_function_calls([(server.make_request, ("POST", "/completion", dict(body))) for _ in range(N)])
    assert all(r.status_code == 200 for r in results), [(r.status_code, r.body) for r in results]
    for r in results:
        assert r.body["tokens_predicted"] == 24, r.body


def test_negative_value_is_refused_at_startup():
    global server
    server.extra_args = ["--kv-admission", "--kv-admission-decode-tokens", "-1"]
    with pytest.raises(Exception):
        server.start(timeout_seconds=10)
