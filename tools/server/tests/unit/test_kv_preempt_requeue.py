import pytest
from utils import ServerError, ServerPreset, parallel_function_calls, use_local_stories260k

# Recompute-preemption instead of 503 (row S69-VLLM-W2-RECOMPUTE-PREEMPTION-INSTEAD-OF-503-2026-10-27-01).
# vLLM preempts under KV pressure by freeing the victim's blocks, resetting its computed-token count to 0 and putting it at the
# front of the waiting queue to be recomputed (vllm/v1/core/sched/scheduler.py:1539-1582, sha 924707f1bf94, read, no code copied).
# This server answered the largest request 503 (test_kv_unified_preempt.py). With --kv-preempt-requeue a request that has sent
# NOTHING to the client (a non-streaming one) is put back in the queue and recomputed from its prompt instead; a streaming request
# already sent its headers / first chunk and keeps the 503.
# Same overload as test_kv_unified_preempt.py: 4 ignore_eos generations of 200 tokens against one 256-cell --kv-unified pool.

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
    server.server_metrics = True
    server.kv_unified = True


def _metric(name):
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    assert isinstance(res.body, str)
    for line in res.body.splitlines():
        if line.startswith("llamacpp:%s " % name):
            return float(line.split()[1])
    raise AssertionError("metric llamacpp:%s missing from /metrics" % name)


def _body(**extra):
    body = {"prompt": "Once upon a time", "n_predict": N_PREDICT, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}
    body.update(extra)
    return body


def _run_concurrent(**extra):
    tasks = [(server.make_request, ("POST", "/completion", _body(**extra))) for _ in range(N)]
    return parallel_function_calls(tasks)


def test_flag_off_overload_answers_503():
    # the BEFORE arm: without the flag an overload answers 503 and nothing is requeued
    global server
    server.start()
    results = _run_concurrent()
    assert sum(1 for r in results if r.status_code == 503) >= 1, "control: the overload must preempt someone"
    assert _metric("requests_preempt_requeued_total") == 0


def test_requeue_completes_instead_of_503():
    global server
    server.extra_args = ["--kv-preempt-requeue"]
    server.start()
    results = _run_concurrent()
    assert [r.status_code for r in results] == [200] * N, [r.status_code for r in results]
    for r in results:
        assert r.body["tokens_predicted"] == N_PREDICT
    n_requeued = _metric("requests_preempt_requeued_total")
    assert n_requeued >= 1, "control: the overload must have requeued someone"
    assert _metric("requests_preempted_total") >= n_requeued
    # the decision is visible in the response: the recomputed request says how often it was requeued
    marked = [r for r in results if r.body.get("preempt_requeues", 0) > 0]
    assert len(marked) >= 1
    assert sum(r.body["preempt_requeues"] for r in marked) >= n_requeued


def test_streaming_request_is_not_requeued():
    # negative arm: a streaming request has already sent its 200 headers / first chunk, a recompute could not take that back, so it
    # keeps today's behaviour (error / 503 after preemption) and is NOT requeued
    global server
    server.extra_args = ["--kv-preempt-requeue"]
    server.start()

    def drain():
        # returns the error the client saw: an error chunk inside the stream, or a non-200 answer to the request itself
        try:
            for chunk in server.make_stream_request("POST", "/completion", _body(stream=True)):
                if isinstance(chunk, dict) and "error" in chunk:
                    return chunk["error"]
        except ServerError as e:
            return e.body
        return None

    errors = parallel_function_calls([(drain, ()) for _ in range(N)])
    assert _metric("requests_preempted_total") >= 1, "control: the overload must preempt someone"
    assert _metric("requests_preempt_requeued_total") == 0
    assert any(e is not None for e in errors), "the preempted streaming request must be told, not silently recomputed"


def test_requeue_limit_zero_answers_503():
    # --kv-preempt-requeue-max 0: the flag is on but no request may be requeued, so the old 503 stays
    global server
    server.extra_args = ["--kv-preempt-requeue", "--kv-preempt-requeue-max", "0"]
    server.start()
    results = _run_concurrent()
    assert sum(1 for r in results if r.status_code == 503) >= 1
    assert _metric("requests_preempt_requeued_total") == 0
