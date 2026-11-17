import pytest
from utils import ServerPreset, download_file

# Per-request adapters on /v1/embeddings and /rerank: the same "lora", "lora_adapter", "lora_adapter_scale" fields the
# completion routes take (row S69-ENGREV-LORAX-EMBED-RERANK-PER-REQUEST-LORA-2026-10-27-01).
#
# BEFORE the change both routes ignored the fields: an embedding asked for with the adapter at scale 0 was the same
# vector as one asked for at scale 1 (cosine 1.0), and an adapter id the server does not have answered 200.
# The checks go both ways: scale 1 must equal the server default (the adapter is loaded at scale 1), scale 0 must differ
# from it, and the same request twice must give the same vector, so a difference is the adapter and not noise.

LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"

server = ServerPreset.stories15m_moe()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.lora_files = [download_file(LORA_FILE_URL)]
    server.server_embeddings = True
    server.pooling = 'mean'
    server.n_slots = 2


def _embed(text: str, **extra):
    res = server.make_request("POST", "/v1/embeddings", data={"input": text, **extra})
    return res


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


def test_embedding_lora_per_request_changes_the_vector():
    global server
    server.start()
    text = "Look in thy glass and tell the face thou viewest"

    res_default = _embed(text)
    assert res_default.status_code == 200
    v_default = res_default.body['data'][0]['embedding']

    res_on = _embed(text, lora=[{"id": 0, "scale": 1.0}])
    res_off = _embed(text, lora=[{"id": 0, "scale": 0.0}])
    res_off2 = _embed(text, lora=[{"id": 0, "scale": 0.0}])
    assert res_on.status_code == 200
    assert res_off.status_code == 200
    assert res_off2.status_code == 200
    v_on = res_on.body['data'][0]['embedding']
    v_off = res_off.body['data'][0]['embedding']
    v_off2 = res_off2.body['data'][0]['embedding']
    assert len(v_default) == len(v_on) == len(v_off) > 0

    # control: the same request twice is the same vector, and scale 1 is what the server does by default
    assert _cosine(v_off, v_off2) > 0.999999
    assert _cosine(v_on, v_default) > 0.999999
    # the adapter at scale 0 is a different model: the vector must move (before the change this was 1.0)
    assert _cosine(v_off, v_default) < 0.9999

    # and the default is untouched by the requests that asked for something else
    res_default2 = _embed(text)
    assert res_default2.status_code == 200
    assert _cosine(res_default2.body['data'][0]['embedding'], v_default) > 0.999999


@pytest.mark.parametrize("lora", [
    [{"id": 7, "scale": 1.0}],          # the server loaded one adapter: id 7 does not exist
    [{"id": -1, "scale": 1.0}],
    [{"scale": 1.0}],                   # no id
    [{"id": 0, "scale": "big"}],        # scale is not a number
    {"id": 0, "scale": 1.0},            # not a list
    "0",
])
def test_embedding_lora_invalid_is_refused(lora):
    global server
    server.start()
    res = _embed("hello", lora=lora)
    assert res.status_code == 400
    assert "lora" in res.body['error']['message']
    # control: the same server still answers a well-formed request
    assert _embed("hello", lora=[{"id": 0, "scale": 0.5}]).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("lora_adapter", 5),                # must be a string
    ("lora_adapter_scale", "big"),      # must be a number
])
def test_embedding_lora_adapter_fields_are_type_checked(field, value):
    global server
    server.start()
    res = _embed("hello", **{field: value})
    assert res.status_code == 400
    assert field in res.body['error']['message']


def test_rerank_lora_invalid_is_refused():
    # /rerank reads the same fields. No adapter exists for the tiny reranker fixture, so this checks that the route
    # parses and validates them (before the change a wrong id answered 200); the path that applies an adapter to a
    # rerank task is the one the embedding test above exercises.
    global server
    server = ServerPreset.jina_reranker_tiny()
    server.start()
    body = {"query": "Machine learning is", "documents": ["A machine is a physical system.", "Paris is in France."]}
    res_ok = server.make_request("POST", "/rerank", data=body)
    assert res_ok.status_code == 200
    res_bad = server.make_request("POST", "/rerank", data={**body, "lora": [{"id": 0, "scale": 1.0}]})
    assert res_bad.status_code == 400   # this server loaded no adapter at all: id 0 is out of range
    assert "lora" in res_bad.body['error']['message']
    res_bad2 = server.make_request("POST", "/rerank", data={**body, "lora": "0"})
    assert res_bad2.status_code == 400
    res_empty = server.make_request("POST", "/rerank", data={**body, "lora": []})
    assert res_empty.status_code == 200  # an empty list selects nothing and is not an error
