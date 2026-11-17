import json
import requests
import pytest
from utils import *

# Ollama-native API translation shim (row S69-SERVING-SURFACE-OLLAMA-NATIVE-API-SHIM-2026-09-27-01).
# fail-before: every /api/* route below returns 404 today (verified by reading server.cpp's route
# table before this row's code existed -- grep found zero /api/(chat|generate|embed|tags|show|ps|
# version) registrations, "/v1/models" used as a positive control). pass-after: this file.

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def ndjson_lines(server: ServerProcess, path: str, data: dict):
    """POSTs to an ollama-shaped NDJSON streaming endpoint and yields each parsed JSON line.
    Ollama's NDJSON framing is one bare JSON object per line (no "data: " SSE prefix, no
    "[DONE]" sentinel -- the LAST line's own `done: true` field is the end-of-stream signal),
    which is why this does not reuse utils.py's make_stream_request (that helper is SSE-specific:
    it strips a "data: " prefix and breaks on a literal "[DONE]" line, per its own source at
    tools/server/tests/utils.py:430-437)."""
    url = f"http://{server.server_host}:{server.server_port}{path}"
    response = requests.post(url, json=data, stream=True, timeout=DEFAULT_REQUEST_TIMEOUT)
    assert response.status_code == 200, response.text
    assert response.headers.get("content-type", "").startswith("application/x-ndjson")
    for line_bytes in response.iter_lines():
        if not line_bytes:
            continue
        yield json.loads(line_bytes.decode("utf-8"))


def test_ollama_chat_nonstream():
    global server
    server.start()
    res = server.make_request("POST", "/api/chat", data={
        "model": "tinyllama-2",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    })
    assert res.status_code == 200
    # utils.ServerResponse.headers is a plain dict holding the server's "Content-Type" spelling, so look it up
    # case-insensitively (a lowercase .get() read '' and failed against a correct application/json header).
    content_type = {k.lower(): v for k, v in res.headers.items()}.get("content-type", "")
    assert content_type.startswith("application/json")
    assert res.body["done"] is True
    assert res.body["model"] == "tinyllama-2"
    assert "message" in res.body
    assert res.body["message"]["role"] == "assistant"
    assert isinstance(res.body["message"]["content"], str)
    assert len(res.body["message"]["content"]) > 0
    assert res.body["eval_count"] > 0
    assert res.body["prompt_eval_count"] > 0


def test_ollama_chat_stream():
    global server
    server.start()
    lines = list(ndjson_lines(server, "/api/chat", {
        "model": "tinyllama-2",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }))
    # at least one content chunk plus the final done chunk
    assert len(lines) >= 2
    for line in lines[:-1]:
        assert line["done"] is False
        assert "message" in line
    last = lines[-1]
    assert last["done"] is True
    assert last["done_reason"] == "stop"
    assert last["eval_count"] > 0
    assert last["prompt_eval_count"] > 0
    assert last["total_duration"] > 0


def test_ollama_generate_nonstream():
    global server
    server.start()
    res = server.make_request("POST", "/api/generate", data={
        "model": "tinyllama-2",
        "prompt": "hello",
        "stream": False,
    })
    assert res.status_code == 200
    assert res.body["done"] is True
    assert isinstance(res.body["response"], str)
    assert len(res.body["response"]) > 0
    assert res.body["eval_count"] > 0


def test_ollama_generate_stream():
    global server
    server.start()
    lines = list(ndjson_lines(server, "/api/generate", {
        "model": "tinyllama-2",
        "prompt": "hello",
        "stream": True,
    }))
    assert len(lines) >= 2
    for line in lines[:-1]:
        assert line["done"] is False
        assert "response" in line
    last = lines[-1]
    assert last["done"] is True
    assert last["eval_count"] > 0


def test_ollama_generate_empty_prompt_loads_only():
    global server
    server.start()
    res = server.make_request("POST", "/api/generate", data={
        "model": "tinyllama-2",
        "prompt": "",
        "stream": False,
    })
    assert res.status_code == 200
    assert res.body["done"] is True
    assert res.body["done_reason"] == "load"
    assert res.body["response"] == ""


def test_ollama_embed():
    global server
    server.pooling = "mean"
    server.server_embeddings = True
    server.start()
    res = server.make_request("POST", "/api/embed", data={
        "model": "tinyllama-2",
        "input": "hello world",
    })
    assert res.status_code == 200
    assert res.body["model"] == "tinyllama-2"
    assert len(res.body["embeddings"]) == 1
    assert len(res.body["embeddings"][0]) > 1
    assert res.body["prompt_eval_count"] > 0


def test_ollama_embed_dimensions_passthrough():
    # cross-checks TASK A (row S69-SERVING-SURFACE-EMBED-DIMENSIONS-2026-09-27-01) is reachable
    # through the ollama-native route too, not just /v1/embeddings directly.
    global server
    server.pooling = "mean"
    server.server_embeddings = True
    server.start()
    res_native = server.make_request("POST", "/api/embed", data={
        "model": "tinyllama-2", "input": "hello world",
    })
    native_len = len(res_native.body["embeddings"][0])
    assert native_len > 8

    res = server.make_request("POST", "/api/embed", data={
        "model": "tinyllama-2", "input": "hello world", "dimensions": 8,
    })
    assert res.status_code == 200
    assert len(res.body["embeddings"][0]) == 8


def test_ollama_legacy_embeddings_shape():
    # older /api/embeddings answers {"embedding": [floats]} -- ONE flat vector. RESEARCH-DONE:
    # https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md "Generate Embedding". It was routed to
    # the /api/embed handler and answered {"embeddings": [[..]]}.
    global server
    server.pooling = "mean"
    server.server_embeddings = True
    server.start()
    res = server.make_request("POST", "/api/embeddings", data={
        "model": "tinyllama-2", "prompt": "hello world",
    })
    assert res.status_code == 200
    assert "embeddings" not in res.body
    vec = res.body["embedding"]
    assert len(vec) > 1 and all(isinstance(x, float) for x in vec)
    # the same text through /api/embed gives the same vector -- only the envelope differs
    res_new = server.make_request("POST", "/api/embed", data={
        "model": "tinyllama-2", "input": "hello world",
    })
    assert res_new.status_code == 200
    assert res_new.body["embeddings"][0] == vec


def test_ollama_embed_missing_input_is_400():
    global server
    server.server_embeddings = True
    server.start()
    res = server.make_request("POST", "/api/embed", data={"model": "tinyllama-2"})
    assert res.status_code == 400
    assert "error" in res.body


def test_ollama_tags():
    global server
    server.start()
    res = server.make_request("GET", "/api/tags", data=None)
    assert res.status_code == 200
    assert "models" in res.body
    assert len(res.body["models"]) == 1
    model = res.body["models"][0]
    assert model["name"].endswith(":latest")
    assert model["digest"].startswith("sha256:")
    assert len(model["digest"]) == len("sha256:") + 64


def test_ollama_show():
    global server
    server.start()
    res = server.make_request("POST", "/api/show", data={"name": "tinyllama-2"})
    assert res.status_code == 200
    assert "details" in res.body
    assert res.body["details"]["format"] == "gguf"
    assert "capabilities" in res.body


def test_ollama_show_missing_name_is_400():
    global server
    server.start()
    res = server.make_request("POST", "/api/show", data={})
    assert res.status_code == 400
    assert "error" in res.body


def test_ollama_ps():
    global server
    server.start()
    res = server.make_request("GET", "/api/ps", data=None)
    assert res.status_code == 200
    assert "models" in res.body
    assert len(res.body["models"]) == 1
    assert "expires_at" in res.body["models"][0]


def test_ollama_version():
    global server
    server.start()
    res = server.make_request("GET", "/api/version", data=None)
    assert res.status_code == 200
    assert "version" in res.body
    assert isinstance(res.body["version"], str)
    assert len(res.body["version"]) > 0
