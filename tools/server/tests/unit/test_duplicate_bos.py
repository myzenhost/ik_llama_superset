# A raw /completion prompt that already starts with the BOS string gets one BOS, not two
# (row S69-OLLAMA-PICK-DOUBLE-BOS-STRIP-RAW-PROMPTS-2026-09-29-01).
#
# SOURCE-READ: ollama/ollama a8aaf9fcfad23dc8d6f07b8109b1ba5ed310d76c llm/llama_server.go:246-280 (MIT): a raw prompt that
# starts with the BOS string loses it when the tokenizer adds BOS. Reimplemented after tokenizing, no code copied.
#
# Model: stories260K (BOS "<s>", add_bos on). The server echoes the tokenized prompt as text in body["prompt"], so the
# number of "<s>" in it is the number of BOS tokens the model saw. Controls: no prefix -> one BOS (unchanged); /tokenize
# keeps both (the exclusion documented in the row); a token-id array prompt is sent as given (the way to send two on purpose).
import pytest

from utils import ServerPreset, use_local_stories260k  # noqa: E402

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 1


def _complete(prompt):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": 2, "temperature": 0.0, "cache_prompt": False})
    assert res.status_code == 200
    return res.body


def test_prompt_starting_with_the_bos_string_gets_one_bos():
    global server
    server.start()
    body = _complete("<s>Once upon a time")
    assert body["prompt"].count("<s>") == 1, body["prompt"]


def test_prompt_with_two_leading_bos_strings_loses_one():
    global server
    server.start()
    body = _complete("<s><s>Once upon a time")
    assert body["prompt"].count("<s>") == 2, "control: only the automatic BOS is removed, an explicit second one stays: " + body["prompt"]


def test_plain_prompt_is_unchanged():
    global server
    server.start()
    body = _complete("Once upon a time")
    assert body["prompt"].count("<s>") == 1, body["prompt"]
    assert body["tokens_evaluated"] == 5


def test_tokenize_endpoint_is_not_changed():
    global server
    server.start()
    res = server.make_request("POST", "/tokenize", data={"content": "<s>Once upon a time", "add_special": True, "parse_special": True})
    assert res.status_code == 200
    toks = res.body["tokens"]
    assert toks[0] == toks[1], "control: /tokenize still reports the double BOS, so the strip is a /completion-path change"


def test_token_id_array_is_sent_as_given():
    global server
    server.start()
    res = server.make_request("POST", "/tokenize", data={"content": "<s>Once upon a time", "add_special": True, "parse_special": True})
    toks = res.body["tokens"]
    assert toks[0] == toks[1]
    body = _complete(toks)
    assert body["prompt"].count("<s>") == 2, body["prompt"]
