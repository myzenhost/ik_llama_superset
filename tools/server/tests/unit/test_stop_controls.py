# min_tokens and stop_token_ids request fields (row S69-VLLM-PICK-REQUEST-STOP-CONTROLS-2026-09-29-01).
#
# Idea and semantics from vLLM SamplingParams (engines/vllm vllm/sampling_params.py:269-280, :618-625 and
# v1/sample/logits_processor/builtin.py:165-330, Apache-2.0); reimplemented, no code copied.
#
#   min_tokens      the end-of-generation tokens and stop_token_ids are masked until that many tokens exist
#   stop_token_ids  sampling one of these ends the generation like EOS; the token is not in the output
#
# Model: stories260K (n_vocab 512), read from the local HF cache like test_control_vectors.py. The EOS id is
# read from the gguf, not assumed. Every expectation has a control that shows the field is what changed the result.
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "gguf-py"))
import gguf  # noqa: E402

from utils import ServerPreset, local_model_path, use_local_stories260k  # noqa: E402

server = ServerPreset.tinyllama2()

_MODEL_GGUF = local_model_path("ggml-org/test-model-stories260K", "stories260K-f32.gguf")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 1


def _eos_id() -> int:
    reader = gguf.GGUFReader(_MODEL_GGUF)
    return int(reader.fields["tokenizer.ggml.eos_token_id"].parts[-1][0])


def _complete(srv, **extra):
    data = {
        "prompt": "Once upon a time",
        "n_predict": 32,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "return_tokens": True,
    }
    data.update(extra)
    return srv.make_request("POST", "/completion", data=data)


def test_min_tokens_holds_back_eos():
    global server
    server.start()
    eos = _eos_id()
    bias = [[eos, 100.0]]  # makes EOS the clear first choice at every position

    base = _complete(server, logit_bias=bias)
    assert base.status_code == 200
    n_base = len(base.body["tokens"])
    assert n_base < 10, "control: with EOS boosted and no min_tokens the run must end almost at once"
    assert base.body["stop_type"] == "eos"

    held = _complete(server, logit_bias=bias, min_tokens=10)
    assert held.status_code == 200
    assert eos not in held.body["tokens"][:10], "no EOS in the first min_tokens outputs"
    assert len(held.body["tokens"]) >= 10
    assert held.body["stop_type"] == "eos", "after min_tokens the boosted EOS ends the run"


def test_min_tokens_above_n_predict_is_refused():
    global server
    server.start()
    res = _complete(server, n_predict=5, min_tokens=10)
    assert res.status_code == 400
    assert "min_tokens" in str(res.body)


def test_min_tokens_is_applied_under_a_grammar_that_allows_other_tokens():
    """With a grammar that allows both more letters and the end, the mask still holds EOS back for min_tokens tokens."""
    global server
    server.start()
    eos = _eos_id()
    bias = [[eos, 100.0]]
    grammar = 'root ::= [a-z ]+'

    base = _complete(server, logit_bias=bias, grammar=grammar, n_predict=40)
    assert base.status_code == 200
    assert len(base.body["tokens"]) < 10, "control: without min_tokens the boosted EOS ends the run almost at once"

    held = _complete(server, logit_bias=bias, grammar=grammar, n_predict=40, min_tokens=10)
    assert held.status_code == 200
    assert len(held.body["tokens"]) >= 10
    assert eos not in held.body["tokens"][:10]
    assert all(c in "abcdefghijklmnopqrstuvwxyz " for c in held.body["content"]), "every held-back token still obeys the grammar"


def test_min_tokens_under_a_grammar_that_can_only_end_terminates_by_restoring_the_stop_logits():
    """The grammar allows nothing after "ab" but the end; the masked EOS is restored (vLLM does the same) and the run ends."""
    global server
    server.start()
    res = _complete(server, grammar='root ::= "ab"', n_predict=16, min_tokens=10)
    assert res.status_code == 200
    assert res.body["content"] == "ab"
    assert res.body["stop_type"] == "eos"
    # control: the same request without min_tokens gives the same text, so the restore changed nothing but allowed the end
    plain = _complete(server, grammar='root ::= "ab"', n_predict=16)
    assert plain.body["content"] == "ab"


def test_stop_token_ids_ends_generation_and_hides_the_token():
    global server
    server.start()
    base = _complete(server, n_predict=12)
    assert base.status_code == 200
    toks = base.body["tokens"]
    assert len(toks) >= 6, "control: need a few greedy tokens to pick from"

    # the first position >= 2 whose token has not appeared before, so stopping there is unambiguous
    idx = next(i for i in range(2, len(toks)) if toks[i] not in toks[:i])
    stop_id = toks[idx]

    stopped = _complete(server, n_predict=12, stop_token_ids=[stop_id])
    assert stopped.status_code == 200
    assert stopped.body["tokens"] == toks[:idx], "generation must end right before the stop token, which is not emitted"
    assert stopped.body["stop_type"] == "eos"
    assert base.body["content"].startswith(stopped.body["content"])
    assert len(stopped.body["content"]) < len(base.body["content"])

    # control: a token that never occurs leaves the output untouched
    unused = next(t for t in range(3, 512) if t not in toks)
    same = _complete(server, n_predict=12, stop_token_ids=[unused])
    assert same.body["tokens"] == toks


def _stop_string_from(content: str) -> str:
    """A 3-character piece of the greedy output, taken from position 8 on, that does not occur earlier in the text."""
    for i in range(8, len(content) - 3):
        piece = content[i:i + 3]
        if content.find(piece) == i and not piece.isspace():
            return piece
    raise AssertionError("control: no usable stop string in " + repr(content))


def test_include_stop_str_in_output_keeps_the_stop_string():
    # SOURCE-READ: engines/vllm/vllm/v1/engine/detokenizer.py:87,:107 (stop string is cut unless the flag is set).
    global server
    server.start()
    base = _complete(server, n_predict=40)
    assert base.status_code == 200
    text = base.body["content"]
    stop = _stop_string_from(text)
    idx = text.find(stop)

    cut = _complete(server, n_predict=40, stop=[stop])
    assert cut.status_code == 200
    assert cut.body["content"] == text[:idx], "default: the stop string is not in the output"
    assert cut.body["stop_type"] == "word"

    kept = _complete(server, n_predict=40, stop=[stop], include_stop_str_in_output=True)
    assert kept.status_code == 200
    assert kept.body["content"] == text[:idx + len(stop)], "with the flag: the output ends with the stop string"
    assert kept.body["content"].endswith(stop)
    assert kept.body["stop_type"] == "word"

    # control: a stop string that never occurs leaves the output untouched, flag or not
    absent = _complete(server, n_predict=40, stop=["☃☃"], include_stop_str_in_output=True)
    assert absent.body["content"] == text


def test_include_stop_str_in_output_streaming_matches_non_streaming():
    global server
    server.start()
    text = _complete(server, n_predict=40).body["content"]
    stop = _stop_string_from(text)
    idx = text.find(stop)

    data = {"prompt": "Once upon a time", "n_predict": 40, "temperature": 0.0, "seed": 42, "cache_prompt": False,
            "stop": [stop], "include_stop_str_in_output": True, "stream": True}
    streamed = "".join(chunk.get("content", "") for chunk in server.make_stream_request("POST", "/completion", data=data))
    assert streamed == text[:idx + len(stop)]


def test_stop_token_ids_out_of_vocabulary_is_refused():
    global server
    server.start()
    res = _complete(server, stop_token_ids=[10**6])
    assert res.status_code == 400
    res = _complete(server, stop_token_ids="not a list")
    assert res.status_code == 400
