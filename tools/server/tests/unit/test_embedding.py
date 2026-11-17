import base64
import math
import struct
import pytest
from openai import OpenAI
from utils import *

server = ServerPreset.bert_bge_small()

EPSILON = 1e-3

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.bert_bge_small()


def test_embedding_single():
    global server
    server.pooling = 'last'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": "I believe the meaning of life is",
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 1
    assert 'embedding' in res.body['data'][0]
    assert len(res.body['data'][0]['embedding']) > 1

    # make sure embedding vector is normalized
    assert abs(sum([x ** 2 for x in res.body['data'][0]['embedding']]) - 1) < EPSILON


def test_embedding_multiple():
    global server
    server.pooling = 'last'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": [
            "I believe the meaning of life is",
            "Write a joke about AI from a very long prompt which will not be truncated",
            "This is a test",
            "This is another test",
        ],
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 4
    for d in res.body['data']:
        assert 'embedding' in d
        assert len(d['embedding']) > 1


def test_embedding_multiple_with_fa():
    server = ServerPreset.bert_bge_small_with_fa()
    server.pooling = 'last'
    server.start()
    # one of these should trigger the FA branch (i.e. context size % 256 == 0)
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": [
            "a "*253,
            "b "*254,
            "c "*255,
            "d "*256,
        ],
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 4
    for d in res.body['data']:
        assert 'embedding' in d
        assert len(d['embedding']) > 1


@pytest.mark.parametrize(
    "input,is_multi_prompt",
    [
        # do not crash on empty input
        ("", False),
        # single prompt
        ("string", False),
        ([12, 34, 56], False),
        ([12, 34, "string", 56, 78], False),
        # multiple prompts
        (["string1", "string2"], True),
        (["string1", [12, 34, 56]], True),
        ([[12, 34, 56], [12, 34, 56]], True),
        ([[12, 34, 56], [12, "string", 34, 56]], True),
        # object entries
        ({"prompt_string": "string"}, False),
        ({"content": [{"type": "text", "text": "string"}]}, False),
        (["string1", {"prompt_string": "string2"}, {"content": [{"type": "text", "text": "string3"}]}], True),
    ]
)
def test_embedding_mixed_input(input, is_multi_prompt: bool):
    global server
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={"input": input})
    assert res.status_code == 200
    data = res.body['data']
    if is_multi_prompt:
        assert len(data) == len(input)
        for d in data:
            assert 'embedding' in d
            assert len(d['embedding']) > 1
    else:
        assert 'embedding' in data[0]
        assert len(data[0]['embedding']) > 1


def test_embedding_content_text_same_as_string():
    global server
    server.pooling = 'last'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": [
            "hello world",
            {"content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]},
        ],
    })
    assert res.status_code == 200
    data = res.body['data']
    assert data[0]['embedding'] == data[1]['embedding']


@pytest.mark.parametrize(
    "input",
    [
        [],
        {"content": "string"},
        {"content": [{"type": "unknown"}]},
        # model is not multimodal
        {"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
        {"content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]},
        {"content": [{"type": "input_video", "input_video": {"url": "data:video/mp4;base64,AAAA"}}]},
    ]
)
def test_embedding_invalid_input(input):
    global server
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={"input": input})
    assert res.status_code != 200


def test_embedding_pooling_mean():
    global server
    server.pooling = 'mean'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": "I believe the meaning of life is",
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 1
    assert 'embedding' in res.body['data'][0]
    assert len(res.body['data'][0]['embedding']) > 1

    # make sure embedding vector is normalized
    assert abs(sum([x ** 2 for x in res.body['data'][0]['embedding']]) - 1) < EPSILON


def test_embedding_pooling_mean_multiple():
    global server
    server.pooling = 'mean'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": [
            "I believe the meaning of life is",
            "Write a joke about AI",
            "This is a test",
        ],
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 3
    for d in res.body['data']:
        assert 'embedding' in d
        assert len(d['embedding']) > 1


def test_embedding_pooling_none():
    global server
    server.pooling = 'none'
    server.start()
    res = server.make_request("POST", "/embeddings", data={
        "input": "hello hello hello",
    })
    assert res.status_code == 200
    assert 'embedding' in res.body[0]
    assert len(res.body[0]['embedding']) == 5 # 3 text tokens + 2 special

    # make sure embedding vector is not normalized
    for x in res.body[0]['embedding']:
        assert abs(sum([x ** 2 for x in x]) - 1) > EPSILON


def test_embedding_pooling_none_oai():
    global server
    server.pooling = 'none'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": "hello hello hello",
    })

    # /v1/embeddings does not support pooling type 'none'
    assert res.status_code == 400
    assert "error" in res.body


def test_embedding_openai_library_single():
    global server
    server.pooling = 'last'
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.embeddings.create(model="text-embedding-3-small", input="I believe the meaning of life is")
    assert len(res.data) == 1
    assert len(res.data[0].embedding) > 1


def test_embedding_openai_library_multiple():
    global server
    server.pooling = 'last'
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.embeddings.create(model="text-embedding-3-small", input=[
        "I believe the meaning of life is",
        "Write a joke about AI from a very long prompt which will not be truncated",
        "This is a test",
        "This is another test",
    ])
    assert len(res.data) == 4
    for d in res.data:
        assert len(d.embedding) > 1


def test_embedding_error_prompt_too_long():
    global server
    server.pooling = 'last'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": "This is a test " * 512,
    })
    assert res.status_code != 200
    assert "too large" in res.body["error"]["message"]


def test_embedding_long_text_mean_chunks_and_averages():
    # row S69-ENGINE-MERGE-GPT4ALL-2026-09-27-01: a text longer than one micro-batch is chunked, each chunk embedded, the vectors averaged
    global server
    server.pooling = 'mean'
    server.start()
    long_text = "This is a test " * 512
    res = server.make_request("POST", "/v1/embeddings", data={"input": long_text, "long_text_mode": "mean", "long_text_overlap": 8})
    assert res.status_code == 200
    assert len(res.body['data']) == 1
    vec = res.body['data'][0]['embedding']
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-3  # embd_normalize default (2): the mean is renormalised
    # usage counts every chunk token, so it is at least the text length
    assert res.body['usage']['prompt_tokens'] >= 512
    # a short text is untouched: same vector with and without the field
    a = server.make_request("POST", "/v1/embeddings", data={"input": "This is a test"})
    b = server.make_request("POST", "/v1/embeddings", data={"input": "This is a test", "long_text_mode": "mean"})
    assert a.body['data'][0]['embedding'] == b.body['data'][0]['embedding']
    # the default is still the refusal
    c = server.make_request("POST", "/v1/embeddings", data={"input": long_text, "long_text_mode": "error"})
    assert c.status_code != 200 and "too large" in c.body["error"]["message"]


def test_embedding_long_text_truncate_and_bad_values():
    global server
    server.pooling = 'mean'
    server.start()
    long_text = "This is a test " * 512
    ok = server.make_request("POST", "/v1/embeddings", data={"input": long_text, "long_text_mode": "truncate"})
    assert ok.status_code == 200
    assert ok.body['data'][0]['embedding']
    bad = server.make_request("POST", "/v1/embeddings", data={"input": "x", "long_text_mode": "median"})
    assert bad.status_code == 400
    bad = server.make_request("POST", "/v1/embeddings", data={"input": long_text, "long_text_mode": "mean", "long_text_overlap": 100000})
    assert bad.status_code == 400
    bad = server.make_request("POST", "/v1/embeddings", data={"input": "x", "long_text_mode": "mean", "long_text_overlap": -1})
    assert bad.status_code == 400


def test_same_prompt_give_same_result():
    server.pooling = 'last'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": [
            "I believe the meaning of life is",
            "I believe the meaning of life is",
            "I believe the meaning of life is",
            "I believe the meaning of life is",
            "I believe the meaning of life is",
        ],
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 5
    for i in range(1, len(res.body['data'])):
        v0 = res.body['data'][0]['embedding']
        vi = res.body['data'][i]['embedding']
        for x, y in zip(v0, vi):
            assert abs(x - y) < EPSILON


@pytest.mark.parametrize(
    "content,n_tokens",
    [
        ("I believe the meaning of life is", 9),
        ("This is a test", 6),
    ]
)
def test_embedding_usage_single(content, n_tokens):
    global server
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={"input": content})
    assert res.status_code == 200
    assert res.body['usage']['prompt_tokens'] == res.body['usage']['total_tokens']
    assert res.body['usage']['prompt_tokens'] == n_tokens


def test_embedding_usage_multiple():
    global server
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": [
            "I believe the meaning of life is",
            "I believe the meaning of life is",
        ],
    })
    assert res.status_code == 200
    assert res.body['usage']['prompt_tokens'] == res.body['usage']['total_tokens']
    assert res.body['usage']['prompt_tokens'] == 2 * 9


def test_embedding_openai_library_base64():
    server.start()
    test_input = "Test base64 embedding output"

    # get embedding in default format
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": test_input
    })
    assert res.status_code == 200
    vec0 = res.body["data"][0]["embedding"]

    # get embedding in base64 format
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": test_input,
        "encoding_format": "base64"
    })

    assert res.status_code == 200
    assert "data" in res.body
    assert len(res.body["data"]) == 1

    embedding_data = res.body["data"][0]
    assert "embedding" in embedding_data
    assert isinstance(embedding_data["embedding"], str)

    # Verify embedding is valid base64
    decoded = base64.b64decode(embedding_data["embedding"])
    # Verify decoded data can be converted back to float array
    float_count = len(decoded) // 4  # 4 bytes per float
    floats = struct.unpack(f'{float_count}f', decoded)
    assert len(floats) > 0
    assert all(isinstance(x, float) for x in floats)
    assert len(floats) == len(vec0)

    # make sure the decoded data is the same as the original
    for x, y in zip(floats, vec0):
        assert abs(x - y) < EPSILON


# "dimensions" (Matryoshka/MRL truncation), row S69-SERVING-SURFACE-EMBED-DIMENSIONS-2026-09-27-01.
# Order is truncate-then-renormalize, matching the OpenAI-documented sequence for shortening
# embeddings (https://developers.openai.com/api/docs/guides/embeddings, read 2026-09-27: "When you
# change the dimension manually, you need to be sure to normalize the dimensions of the embedding"
# -- normalization is applied AFTER slicing). ollama's docs/api.md documents the same `dimensions`
# field ("number of dimensions for the embedding") without specifying an order, so the OpenAI
# protocol precedent governs.

def test_embedding_dimensions_truncation():
    global server
    server.pooling = 'mean'
    server.start()

    res_native = server.make_request("POST", "/v1/embeddings", data={
        "input": "I believe the meaning of life is",
    })
    assert res_native.status_code == 200
    native_vec = res_native.body['data'][0]['embedding']
    assert len(native_vec) > 16

    res = server.make_request("POST", "/v1/embeddings", data={
        "input": "I believe the meaning of life is",
        "dimensions": 16,
    })
    assert res.status_code == 200
    vec = res.body['data'][0]['embedding']
    assert len(vec) == 16

    # truncated vector must be re-normalized to unit length
    assert abs(sum([x ** 2 for x in vec]) - 1) < EPSILON

    # truncate-then-normalize: the returned vector must equal the native (already-unit-norm)
    # vector's first 16 dims, rescaled to unit length -- same direction, not independently computed
    head = native_vec[:16]
    head_norm = math.sqrt(sum(x ** 2 for x in head))
    for a, b in zip(vec, head):
        assert abs(a - (b / head_norm)) < EPSILON


def test_embedding_dimensions_omitted_returns_native_size():
    global server
    server.pooling = 'mean'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": "This is a test",
    })
    assert res.status_code == 200
    # unchanged behavior: omitting "dimensions" must still return the model's native size
    assert len(res.body['data'][0]['embedding']) > 16


def test_embedding_dimensions_exceeds_native_returns_400():
    global server
    server.pooling = 'mean'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": "This is a test",
        "dimensions": 999999,
    })
    assert res.status_code == 400
    assert "error" in res.body


def test_embedding_dimensions_non_positive_returns_400():
    global server
    server.pooling = 'mean'
    server.start()
    for bad in (0, -5):
        res = server.make_request("POST", "/v1/embeddings", data={
            "input": "This is a test",
            "dimensions": bad,
        })
        assert res.status_code == 400
        assert "error" in res.body
@pytest.mark.parametrize(
    "data",
    [
        {"input": []},
        {"input": True},
        {"input": "hello", "encoding_format": 1},
    ]
)
def test_embedding_invalid_request(data):
    global server
    server.pooling = 'last'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data=data)
    assert res.status_code == 400
    assert "error" in res.body
    assert res.body["error"]["type"] == "invalid_request_error"
