# Served check of Responses compaction: POST /v1/responses/compact and the compaction_trigger input item.
# Row S69-OLLAMA-PICK-CODEX-RESPONSES-COMPACTION-ENDPOINT-2026-09-29-01.
#
# RESEARCH-DONE: protocol from Ollama openai/responses_compact.go (MIT), re-written in tools/server/server-responses-compact.h;
# the model-free logic (plan, validation, retained set, expansion, trimming, SSE) is covered by tests_cpp/test_server_responses_compact.cpp.
# This file checks what only a served request shows: the routes exist, bad input is a 400, a compaction item in the input
# expands (input_tokens counts the summary and the retained items), and the full summarising call goes through the model.
#
# DETERMINISM-SETUP: stories260K, temperature 0, seed 1. The tiny model cannot be trusted to write a summary, so the success
# assertions use a payload built the way the server builds it (llama_compaction version 1); the live summarising call is checked
# for a well-formed outcome (200 with ONE compaction item, or a clean 5xx saying the call was not valid), never a hang.
import json
import os

import pytest
from utils import ServerPreset, ServerProcess, use_local_stories260k  # noqa: E402

PORT_BASE = int(os.environ.get("LLAMA_SERVER_TEST_PORT_BASE") or os.environ.get("PORT") or "8080")

INPUT = [
    {"type": "message", "role": "user", "content": "Plan a trip to Paris."},
    {"type": "message", "role": "assistant", "content": "Sure, which month?"},
    {"type": "message", "role": "user", "content": "May."},
]


@pytest.fixture()  # function scope: conftest stops every server after each test
def srv():
    s = ServerPreset.tinyllama2()
    use_local_stories260k(s)
    s.n_slots = 1
    s.n_ctx = 2048
    s.jinja = True
    s.chat_template_file = "../../../models/templates/Qwen3.5-4B.jinja"
    s.temperature = 0.0
    s.seed = 1
    s.server_port = PORT_BASE
    s.start()
    yield s
    s.stop()


def _post(s, path, body):
    return s.make_request("POST", path, data=body, timeout=120)


def _payload(summary, retained):
    return json.dumps({"type": "llama_compaction", "version": 1, "summary": summary, "retained": retained})


def test_compact_route_exists_and_validates(srv):
    res = _post(srv, "/v1/responses/compact", {"input": INPUT})  # no model
    assert res.status_code == 400, (res.status_code, res.body)
    assert "model" in json.dumps(res.body)

    res = _post(srv, "/v1/responses/compact", {"model": "m"})  # no input
    assert res.status_code == 400, (res.status_code, res.body)

    res = _post(srv, "/v1/responses/compact", {"model": "m", "input": INPUT + [{"type": "compaction_trigger"}]})
    assert res.status_code == 400, (res.status_code, res.body)
    assert "compaction_trigger" in json.dumps(res.body)

    res = _post(srv, "/responses/compact", {"input": INPUT})  # the unprefixed alias is routed too
    assert res.status_code == 400, (res.status_code, res.body)


def test_trigger_requires_stream(srv):
    res = _post(srv, "/v1/responses", {"model": "m", "input": INPUT + [{"type": "compaction_trigger"}]})
    assert res.status_code == 400, (res.status_code, res.body)
    assert "stream" in json.dumps(res.body)


def test_foreign_compaction_blob_is_refused(srv):
    item = {"type": "compaction", "encrypted_content": "gAAAA-not-ours"}
    res = _post(srv, "/v1/responses/input_tokens", {"model": "m", "input": [item] + INPUT})
    assert res.status_code == 400, (res.status_code, res.body)
    assert "unsupported" in json.dumps(res.body)


def test_compaction_item_expands_in_input_tokens(srv):
    plain = _post(srv, "/v1/responses/input_tokens", {"model": "m", "input": INPUT})
    assert plain.status_code == 200, plain.body
    short = _post(srv, "/v1/responses/input_tokens", {
        "model": "m",
        "input": [{"type": "compaction", "encrypted_content": _payload("tiny", [])}] + INPUT[-1:],
    })
    assert short.status_code == 200, short.body
    long_summary = "The user is planning a trip to Paris and wants to travel in May. " * 12
    long = _post(srv, "/v1/responses/input_tokens", {
        "model": "m",
        "input": [{"type": "compaction", "encrypted_content": _payload(long_summary, [])}] + INPUT[-1:],
    })
    assert long.status_code == 200, long.body
    # the summary is in the prompt: a longer summary counts more tokens, and the expansion adds its call/output pair
    assert long.body["input_tokens"] > short.body["input_tokens"] > 0
    # null run: without the compaction item the long summary is absent, so the count is the small plain one
    assert plain.body["input_tokens"] < long.body["input_tokens"]


def test_compact_runs_the_model_or_fails_cleanly(srv):
    res = _post(srv, "/v1/responses/compact", {"model": "m", "input": INPUT})
    if res.status_code == 200:
        assert res.body["object"] == "response.compaction", res.body
        out = res.body["output"]
        assert len(out) == 1 and out[0]["type"] == "compaction", res.body
        payload = json.loads(out[0]["encrypted_content"])
        assert payload["type"] == "llama_compaction" and payload["version"] == 1
        assert payload["summary"].strip()
    else:
        # the tiny model did not produce one valid create_summary call: the error says so, it does not hang or crash
        assert res.status_code >= 400, res.body
        assert "compaction failed" in json.dumps(res.body) or "create_summary" in json.dumps(res.body), (res.status_code, res.body)
