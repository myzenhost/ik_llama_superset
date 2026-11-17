import time

import pytest
from utils import ServerError, ServerPreset, parallel_function_calls, use_local_stories260k

# Which request a KV-exhaustion preemption evicts (--kv-preempt-victim), row S69-VLLM-W2-PREEMPT-VICTIM-LEAST-PROGRESS-AND-E2E-2026-10-27-01.
# vLLM picks the newest request (FCFS) or the greatest (priority value, arrival) (vllm/v1/core/sched/scheduler.py:759-767, sha 924707f1bf94,
# read, no code copied). This server's default is the largest request (one eviction frees the most room), which is the request that
# started first -- and that is the streaming one, which cannot be requeued (it already sent its 200 headers / first chunk) so its
# client gets the error. --kv-preempt-victim least-progress picks a request that CAN be requeued before a streaming one, and among
# those the one that computed the fewest cells (least work thrown away); --kv-preempt-victim newest is the vLLM FCFS rule.
#
# Workload: ONE streaming request that starts first plus THREE non-streaming ones that start a few ms later, 4 ignore_eos generations of
# 220 tokens against ONE 256-cell --kv-unified pool with --kv-preempt-requeue. Control arm = the default policy: the streaming
# request is the largest, so it is the victim and its client sees the error.

N_PREDICT = 220
N_OTHERS = 3

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_ctx = 256
    server.n_slots = 1 + N_OTHERS
    server.n_batch = 64
    server.server_metrics = True
    server.kv_unified = True


def _metric(name):
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith("llamacpp:%s " % name):
            return float(line.split()[1])
    raise AssertionError("metric llamacpp:%s missing from /metrics" % name)


def _body(**extra):
    body = {"prompt": "Once upon a time", "n_predict": N_PREDICT, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}
    body.update(extra)
    return body


def _errtext(err):
    # the error text of a streamed error chunk or a ServerError body, whichever shape the harness hands over
    if isinstance(err, dict):
        if "message" in err:
            return str(err["message"])
        if "error" in err:
            return _errtext(err["error"])
    return str(err)


def _stream_client():
    # returns (error text or None, tokens_predicted of the final chunk or None)
    err, n = None, None
    try:
        for chunk in server.make_stream_request("POST", "/completion", _body(stream=True)):
            if isinstance(chunk, dict):
                if "error" in chunk:
                    err = chunk["error"]
                if chunk.get("stop"):
                    n = chunk.get("tokens_predicted")
    except ServerError as e:
        err = e.body
    return err, n


def _late_plain_client(delay):
    time.sleep(delay)
    return server.make_request("POST", "/completion", _body())


def _run_mixed():
    tasks = [(_stream_client, ())] + [(_late_plain_client, (0.03 + 0.01 * i,)) for i in range(N_OTHERS)]
    out = parallel_function_calls(tasks)
    return out[0], out[1:]


def _start(*extra):
    server.extra_args = ["--kv-preempt-requeue", "--kv-preempt-requeue-max", "8"] + list(extra)
    server.start()


def test_default_victim_is_the_largest_and_hits_the_stream():
    # control arm, default behaviour unchanged: the streaming request started first, holds the most cells, is evicted and cannot be requeued
    global server
    _start()
    (err, _), plain = _run_mixed()
    assert _metric("requests_preempted_total") >= 1, "control: the overload must preempt someone"
    msg = _errtext(err)
    assert "preempted" in msg, err
    assert "already started streaming" in msg, msg
    assert [r.status_code for r in plain] == [200] * N_OTHERS


@pytest.mark.parametrize("policy", ["least-progress", "newest"])
def test_policy_protects_the_stream_and_requeues_the_rest(policy):
    global server
    _start("--kv-preempt-victim", policy)
    (err, n_stream), plain = _run_mixed()
    # the streaming client got its whole answer, no error chunk
    assert err is None, err
    assert n_stream == N_PREDICT, n_stream
    # the non-streaming requests ended 200 with the whole generation, the evicted ones recomputed
    assert [r.status_code for r in plain] == [200] * N_OTHERS, [r.status_code for r in plain]
    for r in plain:
        assert r.body["tokens_predicted"] == N_PREDICT
    n_requeued = _metric("requests_preempt_requeued_total")
    assert n_requeued >= 1, "control: the overload must have requeued someone"
    assert _metric("requests_preempted_total") == n_requeued, "nobody was answered 503: every preemption was a requeue"
    assert sum(r.body.get("preempt_requeues", 0) for r in plain) == n_requeued
    # the computed work thrown away is exported: each victim held at least its prompt when it was evicted
    assert _metric("kv_preempt_discarded_cells_total") >= n_requeued


def test_policy_without_requeue_flag_answers_503_and_stays_healthy():
    # --kv-preempt-requeue OFF: nobody is requeueable, least-progress degrades to 'fewest cells' and every victim is answered 503;
    # the server stays healthy afterwards
    global server
    server.extra_args = ["--kv-preempt-victim", "least-progress"]
    server.start()
    (err, _), plain = _run_mixed()
    assert _metric("requests_preempted_total") >= 1
    assert _metric("requests_preempt_requeued_total") == 0
    res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 8})
    assert res.status_code == 200, res.body


def test_invalid_policy_is_rejected_at_startup():
    global server
    server.extra_args = ["--kv-preempt-victim", "oldest"]
    with pytest.raises(Exception):
        server.start()
