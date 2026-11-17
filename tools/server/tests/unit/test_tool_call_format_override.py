#!/usr/bin/env python
# Operator override of the XML tool-call format (--tool-call-format-json / --tool-call-format-preset), served end to end.
# Row S69-LOCALAI-PICK-OPERATOR-TOOLCALL-XML-FORMAT-OVERRIDE-2026-09-29-01.
#
# The format below is one NO shipped handler knows: <call name="NAME"><p k="KEY">VALUE</p></call>. With the flag, a request
# that offers a tool gets a parser and a grammar for it, whatever the chat template says. To get deterministic text out of
# the tiny test model the tool declares {"type": "integer", "enum": [7]} (the grammar can only write 7) and the assistant
# turn is prefilled with the opening of the call, so the grammar (fast-forwarded through the prefill) leaves one continuation.
# Startup validation is checked by running the binary with bad flag values: it must refuse them BY NAME before loading a model.
import json
import os
import subprocess
import pytest

from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import ServerPreset, ServerProcess  # noqa: E402

server: ServerProcess | None = None   # set by the autouse fixture below

FMT = ('{"tool_start":"<call name=\\"","tool_sep":"\\">","key_start":"<p k=\\"","key_val_sep":"\\">",'
       '"val_end":"</p>","tool_end":"</call>"}')
TOOL = {"type": "function", "function": {
    "name": "count_things", "description": "Count things.",
    "parameters": {"type": "object",
                   "properties": {"n": {"type": "integer", "enum": [7]}},
                   "required": ["n"]}}}
PREFILL = '<call name="count_things"><p k="n">'
MESSAGES = [{"role": "user", "content": "Count seven things."}, {"role": "assistant", "content": PREFILL}]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_port = 18435          # NOT 8081: that port is the live serving instance on this host
    server.n_slots = 1
    server.n_ctx = 1024
    server.n_predict = 60
    server.jinja = True
    server.chat_template_file = '../../../models/templates/Qwen3-Coder.jinja'


def _chat():
    return server.make_request("POST", "/v1/chat/completions", data={
        "messages": MESSAGES, "tools": [TOOL], "tool_choice": "required", "temperature": 0.0, "max_tokens": 60})


def test_override_finds_the_call():
    server.extra_args = ["--tool-call-format-json", FMT]
    server.start()
    res = _chat()
    assert res.status_code == 200, res.body
    calls = res.body["choices"][0]["message"].get("tool_calls") or []
    assert calls, res.body
    assert calls[0]["function"]["name"] == "count_things", calls[0]
    assert json.loads(calls[0]["function"]["arguments"]) == {"n": 7}, calls[0]


def test_preset_flag_is_accepted_and_serves():
    """qwen3-coder preset on a template that is NOT Qwen's (Hermes writes JSON calls): the flag path (preset -> JSON ->
    request) is wired and overrides the template's own format, so the XML call is found; without it the Hermes parser
    cannot make sense of this text (see the control below, which runs this template too)."""
    server.chat_template_file = '../../../models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja'
    server.extra_args = ["--tool-call-format-preset", "qwen3-coder"]
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "Count seven things."},
                     {"role": "assistant", "content": '<tool_call>\n<function=count_things>\n<parameter=n>\n'}],
        "tools": [TOOL], "tool_choice": "required", "temperature": 0.0, "max_tokens": 60})
    assert res.status_code == 200, res.body
    calls = res.body["choices"][0]["message"].get("tool_calls") or []
    assert calls and json.loads(calls[0]["function"]["arguments"]) == {"n": 7}, res.body


def test_control_without_the_flag_finds_no_call_in_this_format():
    """Control (the behaviour before the override): the Qwen3-Coder handler does not know <call name=...>."""
    server.start()
    res = _chat()
    calls = (res.body["choices"][0]["message"].get("tool_calls") or []) if res.status_code == 200 else []
    assert not calls, f"without the flag the call must not be found: {res.status_code} {res.body}"


@pytest.mark.parametrize("args,needle", [
    (["--tool-call-format-json", '{"bogus":"x"}'], "unknown key 'bogus'"),
    (["--tool-call-format-json", '{"key_start":"<p>","key_val_sep":"=","val_end":"</p>","tool_end":"</c>"}'], "'tool_start'"),
    (["--tool-call-format-json", "not json"], "not valid JSON"),
    (["--tool-call-format-preset", "no-such-preset"], "unknown preset"),
])
def test_bad_flag_is_refused_at_startup_by_name(args, needle):
    binary = os.environ.get("LLAMA_SERVER_BIN_PATH")
    assert binary, "LLAMA_SERVER_BIN_PATH must point at the server binary"
    p = subprocess.run([binary, "-m", "does-not-exist.gguf", "--port", "18436"] + args, capture_output=True, text=True, timeout=60)
    out = p.stdout + p.stderr
    assert p.returncode != 0, out
    assert needle in out, out
