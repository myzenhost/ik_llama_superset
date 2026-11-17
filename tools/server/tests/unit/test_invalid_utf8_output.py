#!/usr/bin/env python
# A model can emit bytes that are not valid UTF-8. Measured 2026-10-01 on a PQ2_0 model (server log line 356:
# "unparsed Content-only output: ...E5 A1 'o'..."): the chat parser refused the text, the non-partial path threw, and
# the server answered HTTP 500 and dropped the whole response. Row
# S69-SERVER-COMPLETION-500-ON-CONTENT-ONLY-PARSE-OF-GARBAGE-OUTPUT-2026-10-01-01.
#
# The tiny test model has byte-fallback tokens, so a huge logit bias on one byte token makes it emit that byte over and
# over: a run of E5 bytes is invalid UTF-8 (a 3-byte lead followed by another lead), which is exactly the shape of the
# measured failure. Fixed behaviour: HTTP 200, the bad bytes arrive as U+FFFD.
import json
import pytest

from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import ServerPreset, ServerProcess  # noqa: E402

server: ServerProcess | None = None   # set by the autouse fixture below

BYTE_E5_TOKEN = 3 + 0xE5   # sentencepiece byte fallback: <0x00> is id 3


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_port = 18432          # NOT 8081: that port is the live serving instance on this host
    server.n_slots = 1
    server.n_ctx = 512


def _chat(**extra):
    data = {
        "messages": [{"role": "user", "content": "Say something."}],
        "max_tokens": 6,
        "temperature": 0.0,
        "logit_bias": [[BYTE_E5_TOKEN, 100.0]],
    }
    data.update(extra)
    return server.make_request("POST", "/v1/chat/completions", data=data)


def test_control_bias_emits_the_invalid_byte_token():
    """Control: the bias lands on the intended byte token (otherwise the other tests prove nothing)."""
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 4, "temperature": 0.0,
        "logit_bias": [[BYTE_E5_TOKEN, 100.0]], "return_tokens": True})
    assert res.status_code == 200, res.body
    assert res.body["tokens"][:4] == [BYTE_E5_TOKEN] * 4, res.body["tokens"]


def test_invalid_utf8_from_model_is_200_not_500():
    server.start()
    res = _chat()
    assert res.status_code == 200, f"invalid UTF-8 output must not fail the request: {res.status_code} {res.body}"
    content = res.body["choices"][0]["message"]["content"]
    assert "�" in content, content
    json.dumps(res.body)  # and the body is valid JSON text


def test_invalid_utf8_streaming_completes():
    """Control: the streaming route never hit the 500 (it parses partial text, which has no throw path), and it still
    finishes cleanly with the fix. NOT asserted: that the held-back tail arrives. The slot keeps text that ends in a
    cut-off multi-byte lead back (server-context.cpp 'incomplete UTF-8 at the end'), and a run of E5 bytes is one
    endless 'cut-off' lead, so a stream made only of those emits no content at all (measured 2026-10-01: one final
    chunk, finish_reason length, empty delta). Filed as its own row."""
    server.start()
    chunks = list(server.make_stream_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "Say something."}],
        "max_tokens": 6, "temperature": 0.0, "stream": True,
        "logit_bias": [[BYTE_E5_TOKEN, 100.0]]}))
    assert chunks, "the stream produced no chunk at all"
    assert not any("error" in c for c in chunks), chunks
    assert chunks[-1]["choices"][0]["finish_reason"] == "length", chunks[-1]
