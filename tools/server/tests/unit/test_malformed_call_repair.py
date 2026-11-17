"""A tool call the model writes without its wrapper tokens is rebuilt by the server (malformed_call_repair).

The model is shown a history in which the assistant wrote the LFM2 call WITHOUT <|tool_call_start|> / <|tool_call_end|> and, at
temperature 0, continues the pattern:   [get_weather(city="Paris")]   (a custom grammar cannot be combined with tools)
  - malformed_call_repair=false (the strict parser): no tool call, the text is returned as content
  - default / true: the call is returned as a tool call get_weather {"city":"Paris"}
  - stream=true: the repair is off in the server (a rewritten tail would not extend the text already sent): content, no tool call

Ollama reads this same output as a call (model/parsers/lfm2.go:154-161). Runs against a LOCAL LFM2 GGUF named by
LLAMA_TEST_HYBRID_MODEL (os.pathsep separated); skipped without it.
"""
import json
import os

import pytest
from utils import ServerProcess

MODELS = [p for p in os.environ.get("LLAMA_TEST_HYBRID_MODEL", "").split(os.pathsep) if p and "lfm" in os.path.basename(p).lower()]
pytestmark = pytest.mark.skipif(not MODELS, reason="set LLAMA_TEST_HYBRID_MODEL to a local LFM2 GGUF")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]
SHOT = '[get_weather(city="London")]'


def _server(model: str) -> ServerProcess:
    s = ServerProcess()
    s.offline = True
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.n_ctx = 2048
    s.n_slots = 1
    s.jinja = True
    s.temperature = 0.0
    s.seed = 1
    return s


def _wrapper_token(s: ServerProcess) -> int:
    res = s.make_request("POST", "/tokenize", data={"content": "<|tool_call_start|>", "parse_special": True})
    assert res.status_code == 200 and len(res.body["tokens"]) == 1, res.body
    return res.body["tokens"][0]


def _chat(s: ServerProcess, **extra):
    body = {
        # the wrapper token is banned, so the model writes the call bare (what a model that forgets the wrapper does)
        "logit_bias": [[_wrapper_token(s), False]],
        # two turns of plain-text history in which the assistant answered with the bare call: a 350M model at temperature 0
        # continues the pattern instead of writing the <|tool_call_start|> wrapper (the observed output is asserted below)
        "messages": [
            {"role": "user", "content": "What is the weather in London?"},
            {"role": "assistant", "content": SHOT},
            {"role": "user", "content": "What is the weather in Paris?"},
        ],
        "tools": TOOLS,
        "max_tokens": 64,
        "temperature": 0,
    }
    body.update(extra)
    return s.make_request("POST", "/v1/chat/completions", data=body)


@pytest.mark.parametrize("model", MODELS)
def test_bare_call_is_rebuilt_unless_switched_off(model: str):
    s = _server(model)
    s.start()
    try:
        off = _chat(s, malformed_call_repair=False)
        assert off.status_code == 200, off.body
        m_off = off.body["choices"][0]["message"]
        assert not m_off.get("tool_calls"), f"strict parser returned a call: {m_off}"
        assert "get_weather" in (m_off.get("content") or ""), m_off

        on = _chat(s)
        assert on.status_code == 200, on.body
        m_on = on.body["choices"][0]["message"]
        calls = m_on.get("tool_calls") or []
        assert len(calls) == 1, f"call not rebuilt: {m_on}"
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
        assert on.body["choices"][0]["finish_reason"] == "tool_calls", on.body["choices"][0]
        print(f"[repair] off: content={m_off.get('content')!r} calls=0 | on: calls={calls[0]['function']}")

        # streaming: the repair is off (a rewritten tail would not extend the text already sent), the text arrives as content
        body = {
            "messages": [
                {"role": "user", "content": "What is the weather in London?"},
                {"role": "assistant", "content": SHOT},
                {"role": "user", "content": "What is the weather in Paris?"},
            ],
            "tools": TOOLS, "max_tokens": 64, "temperature": 0, "stream": True,
            "logit_bias": [[_wrapper_token(s), False]],
        }
        got_calls, text = False, ""
        for chunk in s.make_stream_request("POST", "/v1/chat/completions", data=body):
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                got_calls = got_calls or bool(d.get("tool_calls"))
                text += d.get("content") or ""
        assert not got_calls and "get_weather" in text, f"streamed: calls={got_calls} text={text!r}"
    finally:
        s.stop()
