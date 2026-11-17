import pytest
from utils import ServerPreset, parallel_function_calls, use_local_stories260k

# KV-exhaustion preemption under --kv-unified (row S69-SERVER-KV-UNIFIED-EXHAUSTION-FAILS-ALL-SLOTS-2026-09-27-01).
# 4 concurrent ignore_eos generations of 200 tokens against ONE shared 256-cell pool cannot all fit. Before the fix
# every in-flight request failed together (MEASURED: 4 of 4 HTTP 500 "Context size has been exceeded."). Now the
# largest slot is preempted with a retryable 503 and the rest keep going, so at least one request completes in full.
# Control: the same load WITHOUT --kv-unified (each slot owns a slice) never takes the preemption path.

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


def _run_concurrent():
    body = {"prompt": "Once upon a time", "n_predict": N_PREDICT, "ignore_eos": True, "temperature": 0.0,
            "cache_prompt": False}
    tasks = [(server.make_request, ("POST", "/completion", dict(body))) for _ in range(N)]
    return parallel_function_calls(tasks)


def test_kv_unified_exhaustion_preempts_one_not_all():
    global server
    server.kv_unified = True
    server.start()
    results = _run_concurrent()
    codes = [r.status_code for r in results]
    ok = [r for r in results if r.status_code == 200]
    preempted = [r for r in results if r.status_code == 503]
    # nobody gets the old all-slots 500
    assert 500 not in codes, codes
    # the survivors finish the whole generation
    assert len(ok) >= 1, codes
    for r in ok:
        assert r.body["tokens_predicted"] == N_PREDICT, r.body
    # everyone else was preempted with a retryable, explained error
    assert len(ok) + len(preempted) == N, codes
    for r in preempted:
        assert "preempted" in r.body["error"]["message"], r.body
    # after the storm the server still serves a normal request (no poisoned batch / slot state)
    res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 8})
    assert res.status_code == 200, res.body


def _run_concurrent_with_priority_on_first(prio):
    body = {"prompt": "Once upon a time", "n_predict": N_PREDICT, "ignore_eos": True, "temperature": 0.0,
            "cache_prompt": False}
    tasks = []
    for i in range(N):
        b = dict(body)
        if i == 0:
            b["priority"] = prio
        tasks.append((server.make_request, ("POST", "/completion", b)))
    return parallel_function_calls(tasks)


def test_priority_decides_who_is_preempted():
    # request `priority` (lower = more important): at exhaustion the victim is the slot with the HIGHEST value; equal
    # values fall back to the largest slot. Request 0 is the only one with a priority. Important (-5): it is never the
    # victim while others are active, so it completes in full. Unimportant (+5): it is the first victim, so it is
    # answered 503. Same load, same server flags, only that one field differs.
    global server
    server.kv_unified = True
    server.start()
    important = _run_concurrent_with_priority_on_first(-5)
    assert 500 not in [r.status_code for r in important]
    assert important[0].status_code == 200, (important[0].status_code, important[0].body)
    assert important[0].body["tokens_predicted"] == N_PREDICT
    assert any(r.status_code == 503 for r in important[1:]), "control: the overload must still preempt someone"

    unimportant = _run_concurrent_with_priority_on_first(5)
    assert 500 not in [r.status_code for r in unimportant]
    assert unimportant[0].status_code == 503, (unimportant[0].status_code, unimportant[0].body)
    assert "preempted" in unimportant[0].body["error"]["message"]


def test_kv_admission_prevents_the_exhaustion():
    # Same overload with --kv-admission: requests whose projected peak does not fit are DEFERRED instead of
    # admitted, so nobody is preempted and every request completes in full (serially if it must).
    global server
    server.kv_unified = True
    server.extra_args = ["--kv-admission"]
    server.start()
    results = _run_concurrent()
    assert all(r.status_code == 200 for r in results), [(r.status_code, r.body) for r in results]
    for r in results:
        assert r.body["tokens_predicted"] == N_PREDICT, r.body


def test_per_slot_kv_is_never_preempted():
    global server
    server.kv_unified = False
    server.start()
    results = _run_concurrent()
    assert all(r.status_code == 200 for r in results), [r.status_code for r in results]
