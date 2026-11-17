#!/usr/bin/env python
# Named tool_choice must force THAT tool -- OpenAI {"type":"function","function":{"name":X}}, the Responses API flat
# form {"type":"function","name":X}, and Anthropic {"type":"tool","name":X}. Before 2026-09-28 the OpenAI/Responses
# forms were read as a string, fell to "auto" with only a log warning, and the Anthropic form became "required",
# which let the model call ANY tool. Row S69-REBUILD-IK-LLAMA-SERVER-FOR-TOOL-CHOICE-FIX-01.
#
# Deterministic observable: the rendered prompt. /apply-template and /v1/messages/count_tokens go through the same
# oaicompat_chat_params_parse as generation, so a forced tool must be the ONLY tool the template renders. Each test
# carries a positive control: with "auto", both tools render (proves the template renders tools at all).
import json
import pytest

from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import ServerPreset, ServerProcess  # noqa: E402

server: ServerProcess | None = None   # set by the autouse fixture below

ZETA = {"type": "function", "function": {"name": "zeta_lookup_tool", "description": "Look up a zeta record.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}
OMEGA = {"type": "function", "function": {"name": "omega_render_tool", "description": "Render an omega view.",
         "parameters": {"type": "object", "properties": {"view": {"type": "string"}}, "required": ["view"]}}}
MESSAGES = [{"role": "user", "content": "Do the thing."}]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_port = 18431          # NOT 8081: that port is the live serving instance on this host
    server.n_slots = 1
    server.n_ctx = 4096
    server.jinja = True
    server.chat_template_file = '../../../models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja'


def render(tool_choice):
    res = server.make_request("POST", "/apply-template", data={
        "messages": MESSAGES, "tools": [ZETA, OMEGA], "tool_choice": tool_choice})
    return res


def test_auto_renders_both_tools_control():
    server.start()
    res = render("auto")
    assert res.status_code == 200, res.body
    prompt = res.body["prompt"]
    assert "zeta_lookup_tool" in prompt and "omega_render_tool" in prompt, "control: the template must render both tools"


@pytest.mark.parametrize("tool_choice", [
    {"type": "function", "function": {"name": "omega_render_tool"}},   # Chat Completions
    {"type": "function", "name": "omega_render_tool"},                 # Responses API flat form
])
def test_named_tool_choice_offers_only_that_tool(tool_choice):
    server.start()
    res = render(tool_choice)
    assert res.status_code == 200, res.body
    prompt = res.body["prompt"]
    assert "omega_render_tool" in prompt, "the forced tool must still be offered"
    assert "zeta_lookup_tool" not in prompt, "a forced tool_choice must not offer the other tools"


def test_named_tool_choice_unknown_function_is_refused():
    server.start()
    res = render({"type": "function", "function": {"name": "no_such_tool"}})
    assert res.status_code != 200, "naming a function that is not in tools must be an error, not a silent 'auto'"
    assert "not in tools" in json.dumps(res.body)


def test_anthropic_named_tool_counts_one_tool():
    server.start()
    tools = [{"name": t["function"]["name"], "description": t["function"]["description"],
              "input_schema": t["function"]["parameters"]} for t in (ZETA, OMEGA)]

    def count(tc):
        res = server.make_request("POST", "/v1/messages/count_tokens", data={
            "model": "x", "max_tokens": 16, "messages": MESSAGES, "tools": tools, "tool_choice": tc})
        assert res.status_code == 200, res.body
        return res.body["input_tokens"]

    auto_n = count({"type": "auto"})
    named_n = count({"type": "tool", "name": "omega_render_tool"})
    assert named_n < auto_n, f"Anthropic tool_choice 'tool' must offer one tool: {named_n} tokens vs {auto_n} with auto"
