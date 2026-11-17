import pytest
from utils import ServerPreset, parallel_function_calls, use_local_stories260k

# /metrics counts KV-pressure events (row S69-VLLM-W2-KV-PREEMPT-METRICS-2026-10-27-01). vLLM exposes the number of preempted
# requests next to its KV usage gauge (counter vllm:num_preemptions, vllm/v1/metrics/loggers.py:676-683, sha 924707f1bf94, read, no code copied) so an
# operator can tell "the pool is too small" from "the model is slow". This server already exported kv_cache_usage_ratio; it did
# not say how many requests the exhaustion path had actually failed.
# Same overload as test_kv_unified_preempt.py: 4 ignore_eos generations of 200 tokens against one 256-cell --kv-unified pool.
#  - requests_preempted_total must equal the number of 503 answers (every preemption answers exactly one request);
#  - control: per-slot KV never preempts, the counter exists and stays 0.

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


def _metric(name):
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    assert isinstance(res.body, str)
    for line in res.body.splitlines():
        if line.startswith("llamacpp:%s " % name):
            return float(line.split()[1])
    raise AssertionError("metric llamacpp:%s missing from /metrics" % name)


def _run_concurrent():
    body = {"prompt": "Once upon a time", "n_predict": N_PREDICT, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}
    tasks = [(server.make_request, ("POST", "/completion", dict(body))) for _ in range(N)]
    return parallel_function_calls(tasks)


def test_preempted_counter_matches_503_answers():
    global server
    server.kv_unified = True
    server.start()
    assert _metric("requests_preempted_total") == 0
    results = _run_concurrent()
    n_503 = sum(1 for r in results if r.status_code == 503)
    assert n_503 >= 1, "control: the overload must preempt someone"
    assert _metric("requests_preempted_total") == n_503


def test_per_slot_kv_counter_stays_zero():
    global server
    server.kv_unified = False
    server.start()
    results = _run_concurrent()
    assert all(r.status_code == 200 for r in results)
    assert _metric("requests_preempted_total") == 0
    assert _metric("kv_cache_idle_slot_purges_total") == 0


def test_radix_counters_follow_a_hit(tmp_path):
    # radix_hits_total / radix_hit_tokens_total (server_radix_cache::stats_t, never exported before): request A fills the tree on
    # slot 0, request B (same prompt, other slot) reuses it. Control: a server without --radix-cache exports no radix_* metric.
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tiny_random_model import PROMPT, tiny_server

    def run(radix):
        d = tmp_path / ("on" if radix else "off")
        os.makedirs(str(d), exist_ok=True)
        s = tiny_server(d)
        s.n_slots = 3
        s.server_metrics = True
        s.extra_args = (["--radix-cache"] if radix else ["--no-radix-cache"]) + ["--kv-unified", "--cache-ram", "0"]
        s.start()
        try:
            body = {"prompt": PROMPT, "n_predict": 2, "temperature": 0, "cache_prompt": True}
            assert s.make_request("POST", "/completion", data=dict(body, id_slot=0)).status_code == 200
            assert s.make_request("POST", "/completion", data=dict(body, id_slot=2)).status_code == 200
            return s.make_request("GET", "/metrics").body
        finally:
            s.stop()

    on = run(True)
    assert "llamacpp:radix_hits_total " in on
    hits = float([l for l in on.splitlines() if l.startswith("llamacpp:radix_hits_total ")][0].split()[1])
    toks = float([l for l in on.splitlines() if l.startswith("llamacpp:radix_hit_tokens_total ")][0].split()[1])
    assert hits >= 1 and toks > 40, (hits, toks)
    assert "llamacpp:radix_" not in run(False)
