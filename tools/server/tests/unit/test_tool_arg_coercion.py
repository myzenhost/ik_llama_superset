#!/usr/bin/env python
# Tool-call argument type repair, served end to end. Row S69-KOBOLDCPP-PICK-TOOL-ARG-SCHEMA-TYPE-COERCION-2026-09-29-01.
#
# A model that writes {"n": "5"} for an integer parameter is repaired to {"n": 5} on the final (non-streaming) parse.
# To get a deterministic string out of the tiny test model:
#  - the tool declares {"type": "integer", "enum": ["5"]}: the enum wins in the json-schema grammar, so the grammar can only
#    produce the string "5" while the type the repair looks at is "integer";
#  - the assistant message is PREFILLED with the opening of the call, so the grammar (fast-forwarded through the prefill)
#    leaves exactly one continuation. (Without the prefill the Hermes grammar allows free text before the call and the
#    random model writes a story instead of calling the tool.)
# The same server is asked with tool_arg_coercion false (the pre-change behaviour) as the control, and a streaming
# request must keep the raw text (already-sent deltas cannot be retracted).
import json
import pytest

from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import ServerPreset, ServerProcess  # noqa: E402

server: ServerProcess | None = None   # set by the autouse fixture below

TOOL = {"type": "function", "function": {
    "name": "count_things", "description": "Count things.",
    "parameters": {"type": "object",
                   "properties": {"n": {"type": "integer", "enum": ["5"]}},
                   "required": ["n"]}}}
PREFILL = '<tool_call>\n{"name": "count_things", "arguments": '
MESSAGES = [{"role": "user", "content": "Count five things."}, {"role": "assistant", "content": PREFILL}]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_port = 18433          # NOT 8081: that port is the live serving instance on this host
    server.n_slots = 1
    server.n_ctx = 1024
    server.n_predict = 80
    server.jinja = True
    server.chat_template_file = '../../../models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja'


def _body(**extra):
    data = {"messages": MESSAGES, "tools": [TOOL], "tool_choice": "required", "temperature": 0.0, "max_tokens": 80}
    data.update(extra)
    return data


def _first_call_args(**extra):
    res = server.make_request("POST", "/v1/chat/completions", data=_body(**extra))
    assert res.status_code == 200, res.body
    calls = res.body["choices"][0]["message"].get("tool_calls") or []
    assert calls, res.body
    for c in calls:   # the grammar lets the call repeat until max_tokens; every complete one must agree
        assert c["function"]["name"] == "count_things", c
    return json.loads(calls[0]["function"]["arguments"])


def test_request_field_off_keeps_the_models_string():
    """Control (the behaviour before this change): the call comes back exactly as the grammar made the model write it."""
    server.start()
    assert _first_call_args(tool_arg_coercion=False) == {"n": "5"}


def test_default_repairs_the_type():
    server.start()
    assert _first_call_args() == {"n": 5}


def test_server_flag_off_then_request_on():
    server.extra_args = ["--no-tool-arg-coercion"]
    server.start()
    assert _first_call_args() == {"n": "5"}, "--no-tool-arg-coercion must restore the raw behaviour"
    assert _first_call_args(tool_arg_coercion=True) == {"n": 5}, "the request field wins over the server default"


def test_streaming_keeps_the_raw_text():
    server.start()
    chunks = list(server.make_stream_request("POST", "/v1/chat/completions", data=_body(stream=True)))
    pieces = {}
    for c in chunks:
        for ch in c.get("choices", []):
            for tc in ch.get("delta", {}).get("tool_calls", []) or []:
                pieces.setdefault(tc.get("index", 0), []).append(tc.get("function", {}).get("arguments") or "")
    assert 0 in pieces, chunks
    assert json.loads("".join(pieces[0])) == {"n": "5"}, pieces
