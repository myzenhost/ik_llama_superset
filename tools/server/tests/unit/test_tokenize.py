import pytest
from utils import *
from utils import use_local_stories260k  # explicit, so static checkers see where it comes from

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)


def test_tokenize_detokenize():
    global server
    server.start()
    # tokenize
    content = "What is the capital of France ?"
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content
    })
    assert res_tok.status_code == 200
    assert len(res_tok.body["tokens"]) > 5
    # detokenize
    res_detok = server.make_request("POST", "/detokenize", data={
        "tokens": res_tok.body["tokens"],
    })
    assert res_detok.status_code == 200
    assert res_detok.body["content"].strip() == content


def test_tokenize_with_bos():
    global server
    server.start()
    # tokenize
    content = "What is the capital of France ?"
    bosId = 1
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content,
        "add_special": True,
    })
    assert res_tok.status_code == 200
    assert res_tok.body["tokens"][0] == bosId


def test_tokenize_with_pieces():
    global server
    server.start()
    # tokenize
    content = "This is a test string with unicode 媽 and emoji 🤗"
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content,
        "with_pieces": True,
    })
    assert res_tok.status_code == 200
    for token in res_tok.body["tokens"]:
        assert "id" in token
        assert token["id"] > 0
        assert "piece" in token
        assert len(token["piece"]) > 0


def test_tokenize_parse_special():
    # parse_special defaults to true: "<s>" in the text is the BOS token (id 1 in this vocab).
    # With parse_special false the same text is tokenized as plain characters.
    global server
    server.start()
    content = "<s>Hello"
    res_default = server.make_request("POST", "/tokenize", data={"content": content})
    res_on = server.make_request("POST", "/tokenize", data={"content": content, "parse_special": True})
    res_off = server.make_request("POST", "/tokenize", data={"content": content, "parse_special": False})
    assert res_default.status_code == 200 and res_on.status_code == 200 and res_off.status_code == 200
    assert res_default.body["tokens"] == res_on.body["tokens"]
    assert res_on.body["tokens"][0] == 1
    assert 1 not in res_off.body["tokens"]
    assert len(res_off.body["tokens"]) > len(res_on.body["tokens"])


def test_tokenize_with_offsets():
    # start/stop are code-point offsets into the concatenated pieces: monotone, contiguous, and
    # text[start:stop] == text for every token whose piece is valid unicode.
    global server
    server.start()
    content = "This is a test string with unicode 媽 and emoji 🤗"
    res_plain = server.make_request("POST", "/tokenize", data={"content": content})
    res = server.make_request("POST", "/tokenize", data={"content": content, "with_offsets": True})
    assert res_plain.status_code == 200 and res.status_code == 200
    tokens = res.body["tokens"]
    assert [t["id"] for t in tokens] == res_plain.body["tokens"]
    prev_stop = 0
    full = ""
    for t in tokens:
        assert t["start"] == prev_stop
        assert t["stop"] >= t["start"]
        prev_stop = t["stop"]
        if isinstance(t["text"], str):
            full += t["text"]
    # whole-character tokens: slicing the joined text by the offsets returns each token's text
    if all(isinstance(t["text"], str) for t in tokens):
        for t in tokens:
            assert full[t["start"]:t["stop"]] == t["text"]
        assert prev_stop == len(full)
        assert full.strip() == content


def test_chat_tokenize():
    # /chat_tokenize returns the same prompt as /apply-template plus the tokens of that prompt
    global server
    server.start()
    messages = [{"role": "user", "content": "hi"}]
    res_tpl = server.make_request("POST", "/apply-template", data={"messages": messages})
    res = server.make_request("POST", "/chat_tokenize", data={"messages": messages})
    assert res_tpl.status_code == 200 and res.status_code == 200
    assert res.body["prompt"] == res_tpl.body["prompt"]
    res_tok = server.make_request("POST", "/tokenize", data={"content": res.body["prompt"]})
    assert res.body["tokens"] == res_tok.body["tokens"]
    res_off = server.make_request("POST", "/chat_tokenize", data={"messages": messages, "with_offsets": True})
    assert res_off.status_code == 200
    assert [t["id"] for t in res_off.body["tokens"]] == res.body["tokens"]
    assert all("start" in t and "stop" in t for t in res_off.body["tokens"])


def test_tokenize_with_pieces_bytes():
    # Byte-fallback tokens are not valid UTF-8 on their own, so their piece is an array of byte
    # values. The ids must equal the plain tokenization, and the pieces must reassemble the text.
    global server
    server.start()
    content = "Hello 媽 🤗"
    res_plain = server.make_request("POST", "/tokenize", data={"content": content})
    res_tok = server.make_request("POST", "/tokenize", data={"content": content, "with_pieces": True})
    assert res_plain.status_code == 200 and res_tok.status_code == 200
    tokens = res_tok.body["tokens"]
    assert [t["id"] for t in tokens] == res_plain.body["tokens"]
    raw = b""
    n_byte_pieces = 0
    for t in tokens:
        if isinstance(t["piece"], list):
            assert all(isinstance(b, int) and 0 <= b <= 255 for b in t["piece"])
            raw += bytes(t["piece"])
            n_byte_pieces += 1
        else:
            raw += t["piece"].encode("utf-8")
    assert n_byte_pieces > 0
    assert raw.decode("utf-8").strip() == content
