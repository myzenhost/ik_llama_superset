import os
import threading
import time

import pytest
import requests
from utils import ServerPreset, local_model_path, TMP_DIR

# Router-mode model failover chain (row S69-MLFLOW-PICK-ROUTER-MODE-FAILOVER-CHAIN-2026-10-15-01).
# A preset section may carry `fallback-model = b,c` (or the router --fallback-model for every model). When the model a request asked for
# cannot be made ready, or its child answers 5xx / 429 before any byte of the answer was handed over, the next model of the chain serves
# it; a response that has started is never retried. Contract read from mlflow/gateway/providers/base.py:713-839 (FallbackProvider):
# :762-770 the first provider that does not raise wins, :824-825 `if yielded: raise`, :831-839 the last error is the final error.
#   model-bad : a file that is not a model, so its child cannot load
#   model-s   : a good model with a fallback model-ok
#   model-q   : a good model with --max-queued-requests 1 (a second request is answered 503 by its child) with fallback model-ok
#   model-ok  : a good model

BAD_MODEL_BYTES = b"this is not a gguf file"


def _headers_ci(headers: dict, key: str):
    for k, v in headers.items():
        if k.lower() == key.lower():
            return v
    return None


@pytest.fixture
def router():
    path = local_model_path("ggml-org/test-model-stories260K", "stories260K-f32.gguf")
    if path is None:
        pytest.skip("stories260K-f32.gguf not found in LLAMA_CACHE / HF hub cache / tests/.test_models")
    path = path.replace("\\", "/")
    os.makedirs(TMP_DIR, exist_ok=True)
    bad = os.path.join(TMP_DIR, f"failover_not_a_model_{os.getpid()}.gguf").replace("\\", "/")
    with open(bad, "wb") as f:
        f.write(BAD_MODEL_BYTES)
    ini = os.path.join(TMP_DIR, f"failover_{os.getpid()}.ini")
    with open(ini, "w") as f:
        f.write(
            "[model-bad]\n"
            f"model = {bad}\n"
            "fallback-model = model-ok\n"
            "\n"
            "[model-bad-alone]\n"
            f"model = {bad}\n"
            "\n"
            "[model-s]\n"
            f"model = {path}\n"
            "fallback-model = model-ok\n"
            "\n"
            "[model-q]\n"
            f"model = {path}\n"
            "max-queued-requests = 1\n"
            "fallback-model = model-ok\n"
            "\n"
            "[model-q-alone]\n"
            f"model = {path}\n"
            "max-queued-requests = 1\n"
            "\n"
            "[model-ok]\n"
            f"model = {path}\n"
        )
    server = ServerPreset.router()
    server.models_preset = ini
    server.server_port = 18434  # NOT 8081: that port is the live serving instance on this host
    server.n_ctx = 1024
    server.n_slots = 1
    server.start()
    try:
        yield server
    finally:
        server.stop()
        for p in (ini, bad):
            try:
                os.remove(p)  # SAFE-DELETE-OK: the two files this fixture itself created a few lines above, in tests/tmp
            except OSError:
                pass


def _status(server, model_id):
    res = server.make_request("GET", "/models")
    for item in res.body.get("data", []):
        if item.get("id") == model_id:
            return item["status"]["value"]
    raise AssertionError(f"{model_id} not in /models")


def _chat(server, model, n=4, **extra):
    body = {"model": model, "max_tokens": n, "temperature": 0.0, "messages": [{"role": "user", "content": "hello"}]}
    body.update(extra)
    return server.make_request("POST", "/v1/chat/completions", data=body, timeout=240)


def test_a_model_that_cannot_load_is_replaced_by_its_fallback(router):
    res = _chat(router, "model-bad")
    assert res.status_code == 200, (res.status_code, str(res.body)[:300])
    assert _headers_ci(res.headers, "X-Model-Served") == "model-ok", res.headers
    assert _headers_ci(res.headers, "X-Failover-Attempts") == "2", res.headers
    assert _status(router, "model-ok") == "loaded"


def test_control_without_a_fallback_the_load_failure_is_the_answer(router):
    res = _chat(router, "model-bad-alone")
    assert res.status_code >= 500, (res.status_code, str(res.body)[:300])
    assert "failed to load" in str(res.body), res.body
    assert _headers_ci(res.headers, "X-Model-Served") is None
    assert _status(router, "model-ok") == "unloaded", "nothing may have touched the fallback model"


def _hold_a_model_busy(router, model):
    """a long request on `model` (it takes one request at a time); returns (thread, box) once the model is loaded and generating"""
    box = {}

    def hold():
        box["r"] = router.make_request("POST", "/completion", data={
            "model": model, "prompt": "Once upon a time", "n_predict": 300, "ignore_eos": True, "temperature": 0.0,
            "cache_prompt": False}, timeout=240)

    t = threading.Thread(target=hold)
    t.start()
    end = time.time() + 120
    while time.time() < end and _status(router, model) != "loaded":
        time.sleep(0.05)
    assert _status(router, model) == "loaded"
    time.sleep(0.3)  # the held request is generating
    return t, box


def test_a_child_that_answers_503_before_any_byte_is_replaced_by_the_fallback(router):
    t, held = _hold_a_model_busy(router, "model-q")
    res = _chat(router, "model-q")
    t.join()
    assert res.status_code == 200, (res.status_code, str(res.body)[:300])
    assert _headers_ci(res.headers, "X-Model-Served") == "model-ok", res.headers
    assert held["r"].status_code == 200 and held["r"].body["tokens_predicted"] == 300, "the held request is untouched"


def test_control_without_a_fallback_the_child_503_is_returned_as_it_is(router):
    t, held = _hold_a_model_busy(router, "model-q-alone")
    res = _chat(router, "model-q-alone")
    t.join()
    assert res.status_code == 503, (res.status_code, str(res.body)[:300])
    assert _status(router, "model-ok") == "unloaded"


def test_a_stream_that_dies_after_its_first_chunk_is_not_retried(router):
    psutil = pytest.importorskip("psutil")
    url = f"http://{router.server_host}:{router.server_port}/v1/chat/completions"
    r = requests.post(url, json={
        "model": "model-s", "max_tokens": 3000, "temperature": 0.0, "stream": True,
        "messages": [{"role": "user", "content": "Tell a long story."}]}, stream=True, timeout=240)
    assert r.status_code == 200
    assert _headers_ci(dict(r.headers), "X-Model-Served") == "model-s", "the failover path must have been the one that answered"
    chunks = 0
    killed = False
    try:
        for line in r.iter_lines():
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                chunks += 1
                if chunks == 2 and not killed:
                    for ch in psutil.Process(router.process.pid).children(recursive=True):
                        try:
                            if "model-s" in ch.cmdline():  # one argv token equal to the model name, not a substring of a joined line
                                ch.kill()
                                killed = True
                        except psutil.Error:
                            continue
    except requests.exceptions.RequestException:
        pass  # a dropped connection is one of the legitimate ends of a stream whose model died
    assert killed, "control: the child of model-s must have been found and killed"
    assert chunks < 2900, ("the stream must have been cut, not completed", chunks)
    # the fallback model was never asked: it is still not loaded
    assert _status(router, "model-ok") == "unloaded", "a stream that had started must not be retried on the fallback model"
