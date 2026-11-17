import math

import pytest
from utils import ServerPreset

# A decoder (causal) model used as an embedder, with each pooling type the server offers for it.
#
# With --pooling mean the server used to die while LOADING (ggml.c GGML_ASSERT(ggml_can_mul_mat)): the context reserve
# of the legacy graph builders also builds a "pruned-output" graph (outputs = tokens - 1), a shape pooled embeddings
# never have, because every token of a pooled embedding batch is an output; mean pooling multiplies the per-output
# hidden states by a [tokens, tokens] matrix, so that reserve-only graph did not fit. Upstream llama.cpp has no such
# graph (its reserve is all-outputs and its output-id input is always built), which is why it never showed there.
# "last" and "none" loaded before the fix and are the controls.

EPSILON = 1e-3

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_embeddings = True
    # a pooled sequence must sit whole inside one batch (the server refuses a longer one by name); the preset's 32 is
    # smaller than the longer text below
    server.n_batch = 512


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


@pytest.mark.parametrize("pooling", ["mean", "last"])
def test_embedding_decoder_model_pooling(pooling: str):
    global server
    server.pooling = pooling
    server.start()
    short = "I believe the meaning of life is"
    long = "Write a joke about AI from a very long prompt which will not be truncated"
    res = server.make_request("POST", "/v1/embeddings", data={"input": [short, long]})
    assert res.status_code == 200
    assert len(res.body['data']) == 2
    a = res.body['data'][0]['embedding']
    b = res.body['data'][1]['embedding']
    assert len(a) == len(b) > 0
    assert all(math.isfinite(x) for x in a) and all(math.isfinite(x) for x in b)
    assert abs(_dot(a, a) - 1.0) < EPSILON      # normalised by default
    assert _dot(a, b) < 1.0 - 1e-4             # two different texts are two different vectors
    # one input alone gives the same vector as inside the batch of two: each sequence is pooled on its own
    res1 = server.make_request("POST", "/v1/embeddings", data={"input": short})
    assert res1.status_code == 200
    assert _dot(a, res1.body['data'][0]['embedding']) > 1.0 - 1e-4


def test_embedding_decoder_model_mean_differs_from_last():
    # mean and last are different poolings of the same hidden states: if "mean" silently fell back to "last" the two
    # vectors would be equal
    global server
    text = "Write a joke about AI from a very long prompt which will not be truncated"
    vecs = {}
    for pooling in ("mean", "last"):
        server = ServerPreset.tinyllama2()
        server.server_embeddings = True
        server.n_batch = 512
        server.pooling = pooling
        server.start()
        res = server.make_request("POST", "/v1/embeddings", data={"input": text})
        assert res.status_code == 200
        vecs[pooling] = res.body['data'][0]['embedding']
        server.stop()
    assert _dot(vecs["mean"], vecs["last"]) < 1.0 - 1e-4


def test_embedding_decoder_model_mean_too_long_for_the_batch_is_refused_by_name():
    # the limit that remains is stated, not a crash: a sequence longer than the batch is refused with the batch size
    global server
    server.n_batch = 32
    server.pooling = 'mean'
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": "Write a joke about AI from a very long prompt which will not be truncated"})
    assert res.status_code == 500
    assert "batch size" in res.body['error']['message']
    # and the server is still serving
    res_ok = server.make_request("POST", "/v1/embeddings", data={"input": "hello"})
    assert res_ok.status_code == 200
