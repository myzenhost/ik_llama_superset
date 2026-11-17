"""TGI native API routes, row S69-TGI-PICK-NATIVE-GENERATE-ROUTES-2026-10-18-01.

/generate, /generate_stream, POST / and GET /info speak huggingface/text-generation-inference's wire protocol
(router/src/server.rs, router/src/lib.rs, router/src/validation.rs at b4adbf2f6e2e7212, Apache-2.0).
A server without these routes answers 404, which every test below refuses.
"""
import json

import pytest
import requests
from utils import ServerPreset, use_local_stories260k

server = ServerPreset.tinyllama2()

PROMPT = "Once upon a time"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)  # the cached stories260K; a build without curl cannot fetch it


def _url(path: str) -> str:
    return f"http://{server.server_host}:{server.server_port}{path}"


def _sse(path: str, body: dict) -> list[dict]:
    r = requests.post(_url(path), json=body, stream=True, timeout=60)
    assert r.status_code == 200, r.text
    assert r.headers.get("Content-Type", "").startswith("text/event-stream")
    events = []
    for raw in r.iter_lines():
        line = raw.decode("utf-8")
        if line.startswith("data:"):
            events.append(json.loads(line[5:]))
    return events


def test_generate_is_greedy_with_parameters_and_matches_completion():
    server.start()
    body = {"inputs": PROMPT, "parameters": {"max_new_tokens": 8}}
    a = server.make_request("POST", "/generate", data=body)
    assert a.status_code == 200, a.body
    assert set(a.body.keys()) == {"generated_text"}  # no details unless asked
    b = server.make_request("POST", "/generate", data=body)
    assert b.body["generated_text"] == a.body["generated_text"]  # parameters present, no do_sample -> greedy
    # positive control: the same greedy run through /completion
    c = server.make_request("POST", "/completion", data={"prompt": PROMPT, "n_predict": 8, "temperature": 0.0})
    assert c.status_code == 200
    assert a.body["generated_text"] == c.body["content"]
    assert int(a.headers["x-generated-tokens"]) == 8


def test_generate_details_tokens():
    server.start()
    res = server.make_request("POST", "/generate", data={
        "inputs": PROMPT, "parameters": {"max_new_tokens": 6, "details": True, "top_n_tokens": 3}})
    assert res.status_code == 200, res.body
    d = res.body["details"]
    assert d["finish_reason"] == "length"
    assert d["generated_tokens"] == 6
    assert d["seed"] is None  # greedy
    assert d["prefill"] == []
    assert len(d["tokens"]) == 6
    for t in d["tokens"]:
        assert set(t.keys()) == {"id", "text", "logprob", "special"}
        assert isinstance(t["id"], int) and t["logprob"] <= 0.0
    assert "".join(t["text"] for t in d["tokens"]) == res.body["generated_text"]
    assert len(d["top_tokens"]) == 6 and all(len(x) == 3 for x in d["top_tokens"])
    # greedy: the chosen token is the top candidate
    assert all(d["tokens"][i]["id"] == d["top_tokens"][i][0]["id"] for i in range(6))


def test_compat_route_returns_array_with_full_text():
    server.start()
    res = server.make_request("POST", "/", data={"inputs": PROMPT, "parameters": {"max_new_tokens": 4}})
    assert res.status_code == 200, res.body
    assert isinstance(res.body, list) and len(res.body) == 1
    g = server.make_request("POST", "/generate", data={"inputs": PROMPT, "parameters": {"max_new_tokens": 4}})
    assert res.body[0]["generated_text"] == PROMPT + g.body["generated_text"]


def test_generate_stream_events():
    server.start()
    body = {"inputs": PROMPT, "parameters": {"max_new_tokens": 5, "details": True}}
    events = _sse("/generate_stream", body)
    assert [e["index"] for e in events] == [1, 2, 3, 4, 5]
    assert all(e["generated_text"] is None for e in events[:-1])
    last = events[-1]
    full = server.make_request("POST", "/generate", data=body)
    assert last["generated_text"] == full.body["generated_text"]
    assert "".join(e["token"]["text"] for e in events) == last["generated_text"]
    assert last["details"]["finish_reason"] == "length" and last["details"]["generated_tokens"] == 5
    # compat route with stream: true is the same stream
    compat = _sse("/", {"inputs": PROMPT, "stream": True, "parameters": {"max_new_tokens": 5}})
    assert [e["token"]["id"] for e in compat] == [e["token"]["id"] for e in events]


def test_info():
    server.start()
    res = server.make_request("GET", "/info")
    assert res.status_code == 200, res.body
    assert res.body["model_pipeline_tag"] == "text-generation"
    assert res.body["max_total_tokens"] > 0
    assert res.body["max_input_tokens"] == res.body["max_total_tokens"] - 1
    # best_of fans out as n_cmpl, one slot per sequence: the cap is the slot count (preset --parallel 2)
    assert res.body["max_best_of"] == res.body["max_concurrent_requests"] == 2
    assert res.body["max_stop_sequences"] == 4


@pytest.mark.parametrize("params,needle", [
    ({"temperature": 0.0}, "`temperature` must be strictly positive"),
    ({"top_p": 1.0}, "`top_p` must be > 0.0 and < 1.0"),
    ({"top_n_tokens": 9}, "`top_n_tokens` must be >= 0 and <= 5"),
    ({"stop": ["a", "b", "c", "d", "e"]}, "`stop` supports up to 4 stop sequences"),
    ({"best_of": 2}, "you must use sampling when `best_of` is > 1"),
    ({"best_of": 2, "do_sample": True, "seed": 7}, "`seed` must not be set when `best_of` > 1"),
    ({"best_of": 3, "do_sample": True}, "`best_of` must be > 0 and <= 2. Given: 3"),
    ({"frequency_penalty": 0.5}, "length-normalised"),
    ({"max_new_tokens": "8"}, "`max_new_tokens` must be a number"),
])
def test_validation_is_422(params, needle):
    server.start()
    res = server.make_request("POST", "/generate", data={"inputs": PROMPT, "parameters": params})
    assert res.status_code == 422, res.body
    assert res.body["error_type"] == "validation"
    assert needle in res.body["error"]


def test_stream_validation_error_is_one_event():
    server.start()
    events = _sse("/generate_stream", {"inputs": PROMPT, "parameters": {"temperature": -1}})
    assert len(events) == 1
    assert events[0]["error_type"] == "validation"


def test_stop_sequence_finish_reason():
    server.start()
    g = server.make_request("POST", "/generate", data={"inputs": PROMPT, "parameters": {"max_new_tokens": 16}})
    text = g.body["generated_text"]
    words = [w for w in text.split() if len(w) > 2]
    assert words, text
    stop = words[1] if len(words) > 1 else words[0]
    res = server.make_request("POST", "/generate", data={
        "inputs": PROMPT, "parameters": {"max_new_tokens": 16, "stop": [stop], "details": True}})
    assert res.status_code == 200, res.body
    assert res.body["details"]["finish_reason"] == "stop_sequence"
    assert len(res.body["generated_text"]) < len(text)


# Rows S69-TGI-PICK-DECODER-INPUT-DETAILS-VIA-PROMPT-LOGPROBS-2026-10-19-01 and
# S69-TGI-PICK-BEST-OF-VIA-N-CMPL-RANK-2026-10-19-01 (decline overturned 2026-10-19). Before: both answered 422.

def _tok_pieces(text):
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True, "with_pieces": True})
    assert res.status_code == 200
    return res.body["tokens"]


def _mean(tokens):
    return sum(t["logprob"] for t in tokens) / len(tokens)


def test_decoder_input_details_prefill():
    server.start()
    toks = _tok_pieces(PROMPT)
    res = server.make_request("POST", "/generate", data={
        "inputs": PROMPT, "parameters": {"max_new_tokens": 4, "decoder_input_details": True}})
    assert res.status_code == 200, res.body
    d = res.body["details"]  # details come back with decoder_input_details alone (server.rs:310)
    pre = d["prefill"]
    assert [p["id"] for p in pre] == [t["id"] for t in toks]
    assert [p["text"] for p in pre] == [t["piece"] for t in toks]
    assert pre[0]["logprob"] is None  # the first token has no context
    # the values are the server's prompt_logprobs for the same ids (that field has its own oracle test)
    ref = server.make_request("POST", "/completion", data={
        "prompt": [t["id"] for t in toks], "n_predict": 1, "temperature": 0.0, "prompt_logprobs": True})
    assert ref.status_code == 200
    lps = ref.body["prompt_logprobs"]["logprobs"]
    assert len(lps) == len(pre)
    for p, lp in zip(pre[1:], lps[1:]):
        assert p["logprob"] == pytest.approx(lp, abs=1e-5)
        assert p["logprob"] <= 0.0
    assert len(d["tokens"]) == 4
    # still refused when streaming (server.rs:549-553)
    events = _sse("/generate_stream", {"inputs": PROMPT, "parameters": {"decoder_input_details": True}})
    assert len(events) == 1 and events[0]["error_type"] == "validation"


def test_best_of_picks_highest_mean_logprob():
    server.start()
    res = server.make_request("POST", "/generate", data={
        "inputs": PROMPT, "parameters": {"max_new_tokens": 8, "best_of": 2, "do_sample": True, "temperature": 1.5,
                                         "details": True}})
    assert res.status_code == 200, res.body
    d = res.body["details"]
    others = d["best_of_sequences"]
    assert len(others) == 1
    assert res.body["generated_text"] == "".join(t["text"] for t in d["tokens"])
    # infer/mod.rs:338-344: the answer is the sequence with the highest MEAN generated-token logprob
    for o in others:
        assert _mean(d["tokens"]) >= _mean(o["tokens"])
        assert o["seed"] != d["seed"]  # each n_cmpl child samples with its own seed
        assert o["prefill"] == [] and o["generated_tokens"] == len(o["tokens"])
    # no details asked: only the text, still the best one
    plain = server.make_request("POST", "/generate", data={
        "inputs": PROMPT, "parameters": {"max_new_tokens": 8, "best_of": 2, "do_sample": True}})
    assert plain.status_code == 200 and set(plain.body.keys()) == {"generated_text"}


def test_best_of_with_decoder_input_details_shares_prefill():
    server.start()
    toks = _tok_pieces(PROMPT)
    res = server.make_request("POST", "/generate", data={
        "inputs": PROMPT, "parameters": {"max_new_tokens": 4, "best_of": 2, "do_sample": True,
                                         "decoder_input_details": True}})
    assert res.status_code == 200, res.body
    d = res.body["details"]
    assert [p["id"] for p in d["prefill"]] == [t["id"] for t in toks]
    assert d["prefill"][0]["logprob"] is None and d["prefill"][1]["logprob"] is not None
    assert all(o["prefill"] == d["prefill"] for o in d["best_of_sequences"])
