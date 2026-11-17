import os
import json
import pytest
from utils import ServerPreset, local_model_path

# Regression guard for xinference #5543: a server that streams tool_calls as ABSOLUTE repeats of the arguments so far
# (each chunk re-sends the whole string) breaks every OpenAI client, which concatenates the per-chunk fragments.
# MEASURED 2026-11-16 on this fork (Qwen3-0.6B-Q8_0, CPU, --jinja, tool_choice=required): the tool call arrives as one
# delta {index:0, id, name, arguments:'{"city": "Paris"}'}; no repeat. This test pins that shape: fragments joined must
# be valid JSON equal to the final arguments, and no fragment may re-send an earlier one as a prefix.


def absolute_repeat(fragments):
    """True when some fragment strictly extends the previous non-empty fragment (cumulative/absolute streaming)."""
    prev = ""
    for f in fragments:
        if prev and len(f) > len(prev) and f.startswith(prev):
            return True
        if f:
            prev = f
    return False


def test_detector_flags_absolute_repeats_and_passes_incremental():
    """Negative control: the detector itself must tell the two shapes apart, else the server test below proves nothing."""
    assert absolute_repeat(['{"ci', '{"city": "Pa', '{"city": "Paris"}'])
    assert not absolute_repeat(['{"ci', 'ty": "Pa', 'ris"}'])
    assert not absolute_repeat(['{"city": "Paris"}'])


def _qwen3_path():
    env = os.environ.get("LLAMA_TEST_QWEN3_GGUF")
    if env and os.path.isfile(env):
        return env
    p = local_model_path("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf")
    if p:
        return p
    pytest.skip("Qwen3-0.6B-Q8_0.gguf not found (set LLAMA_TEST_QWEN3_GGUF)")


def test_stream_tool_call_arguments_are_not_absolute_repeats():
    server = ServerPreset.tinyllama2()
    server.server_port = int(os.environ.get("PORT", "19417"))
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = _qwen3_path()
    server.jinja = True
    server.n_ctx = 2048
    server.n_slots = 1
    server.n_predict = 160
    server.temperature = 0.0
    server.start()
    body = {
        "model": "x", "stream": True, "max_tokens": 160, "temperature": 0, "seed": 1, "tool_choice": "required",
        "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    }
    frags, ids, names = [], set(), []
    for chunk in server.make_stream_request("POST", "/v1/chat/completions", data=body):
        for ch in chunk.get("choices", []):
            for tc in ch.get("delta", {}).get("tool_calls") or []:
                assert tc.get("index") == 0
                if tc.get("id"):
                    ids.add(tc["id"])
                fn = tc.get("function", {})
                if fn.get("name"):
                    names.append(fn["name"])
                if fn.get("arguments") is not None:
                    frags.append(fn["arguments"])
    assert frags, "no tool_calls delta streamed"
    assert not absolute_repeat(frags), f"absolute repeats in streamed arguments: {frags}"
    assert len(ids) == 1 and names == ["get_weather"], "id/name must be sent once per call"
    args = json.loads("".join(frags))
    assert isinstance(args, dict) and "city" in args
