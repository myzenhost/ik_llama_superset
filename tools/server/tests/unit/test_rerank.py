import pytest
from utils import *

server = ServerPreset.jina_reranker_tiny()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.jina_reranker_tiny()


TEST_DOCUMENTS = [
    "A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
    "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
    "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
    "Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine."
]


def test_rerank():
    global server
    server.start()
    res = server.make_request("POST", "/rerank", data={
        "query": "Machine learning is",
        "documents": TEST_DOCUMENTS,
    })
    assert res.status_code == 200
    assert len(res.body["results"]) == 4

    most_relevant = res.body["results"][0]
    least_relevant = res.body["results"][0]
    for doc in res.body["results"]:
        if doc["relevance_score"] > most_relevant["relevance_score"]:
            most_relevant = doc
        if doc["relevance_score"] < least_relevant["relevance_score"]:
            least_relevant = doc

    assert most_relevant["relevance_score"] > least_relevant["relevance_score"]
    assert most_relevant["index"] == 2
    assert least_relevant["index"] == 3


@pytest.mark.parametrize("kv_unified", [False, True])
def test_rerank_skips_prompt_cache(kv_unified, tmp_path):
    global server
    server.n_slots = 2
    server.cache_ram = 100
    server.kv_unified = kv_unified
    server.debug = True
    server.log_path = str(tmp_path / "server.log")
    server.start()

    for i in range(4):
        res = server.make_request("POST", "/rerank", data={
            "query": f"query {i}",
            "documents": TEST_DOCUMENTS[i:],
        })
        assert res.status_code == 200

    server.stop()

    with open(server.log_path) as f:
        log = f.read()
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log


def test_rerank_tei_format():
    global server
    server.start()
    res = server.make_request("POST", "/rerank", data={
        "query": "Machine learning is",
        "texts": TEST_DOCUMENTS,
    })
    assert res.status_code == 200
    assert len(res.body) == 4

    most_relevant = res.body[0]
    least_relevant = res.body[0]
    for doc in res.body:
        if doc["score"] > most_relevant["score"]:
            most_relevant = doc
        if doc["score"] < least_relevant["score"]:
            least_relevant = doc

    assert most_relevant["score"] > least_relevant["score"]
    assert most_relevant["index"] == 2
    assert least_relevant["index"] == 3


@pytest.mark.parametrize("documents", [
    [],
    None,
    123,
    [1, 2, 3],
])
def test_invalid_rerank_req(documents):
    global server
    server.start()
    res = server.make_request("POST", "/rerank", data={
        "query": "Machine learning is",
        "documents": documents,
    })
    assert res.status_code == 400
    assert "error" in res.body


# Ids the reference tokenizer gives for the reranker's own tokenizer.json (jinaai/jina-reranker-v1-tiny-en: normalizer
# NFC + Lowercase, pre_tokenizer Whitespace, BPE with unk_token "<unk>"), produced 2026-11-14 with the Hugging Face
# `tokenizers` library, add_special_tokens=False. The GGUF under test is an older conversion tagged model "gpt2" /
# pre "jina-v1-en"; it used to take the GPT-2 path and give <unk> ach ine <unk> learning <unk> is for the first row.
JINA_V1_REFERENCE_IDS = [
    ("Machine learning is", [23448, 22812, 21152]),
    ("machine learning is", [23448, 22812, 21152]),
    ("A machine", [69, 23448]),
    ("Learning is", [22812, 21152]),
    ("Which city?", [21366, 21989, 61]),
    ("Machine learning is ", [23448, 22812, 21152]),
    ("Paris, capitale de la", [27059, 42, 23897, 73, 21171, 21418]),
    ("hello world", [28687, 21628]),
    # an emoji has no token: the reference emits <unk> (id 3) for it and nothing for the spaces around it
    ("Hello, World! café \U0001F999 x", [28687, 42, 21628, 31, 35061, 3, 92]),
]


@pytest.mark.parametrize("text,ids", JINA_V1_REFERENCE_IDS)
def test_rerank_tokenizer_matches_reference(text, ids):
    global server
    server.start()
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": False})
    assert res.status_code == 200
    assert res.body["tokens"] == ids


@pytest.mark.parametrize(
    "query,doc1,doc2,n_tokens",
    [
        # <s> query </s> </s> document </s> per document, with the reference token counts above:
        # (4 + 3 + 2) + (4 + 3 + 2) and (4 + 3 + 3) + (4 + 3 + 6). Upstream's 19 and 26 count the GPT-2-path
        # tokenization of this file, which drops every capital letter (ach ine learning is).
        ("Machine learning is", "A machine", "Learning is", 18),
        ("Which city?", "Machine learning is ", "Paris, capitale de la", 23),
    ]
)
def test_rerank_usage(query, doc1, doc2, n_tokens):
    global server
    server.start()

    res = server.make_request("POST", "/rerank", data={
        "query": query,
        "documents": [
            doc1,
            doc2,
        ]
    })
    assert res.status_code == 200
    assert res.body['usage']['prompt_tokens'] == res.body['usage']['total_tokens']
    assert res.body['usage']['prompt_tokens'] == n_tokens


@pytest.mark.parametrize("top_n,expected_len", [
    (None, len(TEST_DOCUMENTS)),  # no top_n parameter
    (2, 2),
    (4, 4),
    (99, len(TEST_DOCUMENTS)),    # higher than available docs
])
def test_rerank_top_n(top_n, expected_len):
    global server
    server.start()
    data = {
        "query": "Machine learning is",
        "documents": TEST_DOCUMENTS,
    }
    if top_n is not None:
        data["top_n"] = top_n

    res = server.make_request("POST", "/rerank", data=data)
    assert res.status_code == 200
    assert len(res.body["results"]) == expected_len


@pytest.mark.parametrize("top_n,expected_len", [
    (None, len(TEST_DOCUMENTS)),  # no top_n parameter
    (2, 2),
    (4, 4),
    (99, len(TEST_DOCUMENTS)),    # higher than available docs
])
def test_rerank_tei_top_n(top_n, expected_len):
    global server
    server.start()
    data = {
        "query": "Machine learning is",
        "texts": TEST_DOCUMENTS,
    }
    if top_n is not None:
        data["top_n"] = top_n

    res = server.make_request("POST", "/rerank", data=data)
    assert res.status_code == 200
    assert len(res.body) == expected_len

# --- Infinity rerank semantics (row S69-infinity-PICK-RERANK-SIGMOID/-PARAMS-2026-09-28-01) ---
# SOURCE-READ Infinity (MIT): libs/infinity_emb/infinity_emb/inference/batch_handler.py:177-221
# (raw_scores -> sigmoid at :208-210; top_n None or <= 0 -> every result, :218) and
# fastapi_schemas/pymodels.py:221-232 (return_documents / raw_scores / top_n on the generic rerank input).
# DEFAULT DIFFERS ON PURPOSE: Infinity defaults raw_scores=False (sigmoid); this server keeps RAW scores when the
# request does not say, because existing callers were calibrated on raw logits. Sending raw_scores selects it.

import math


def _scores(body):
    return {r["index"]: r["relevance_score"] for r in body["results"]}


def test_rerank_raw_scores_false_is_sigmoid_of_raw():
    global server
    server.start()
    base = {"query": "Machine learning is", "documents": TEST_DOCUMENTS}
    raw = server.make_request("POST", "/rerank", data=dict(base, raw_scores=True))
    sig = server.make_request("POST", "/rerank", data=dict(base, raw_scores=False))
    dflt = server.make_request("POST", "/rerank", data=base)
    assert raw.status_code == sig.status_code == dflt.status_code == 200
    r, s, d = _scores(raw.body), _scores(sig.body), _scores(dflt.body)
    assert d == r                      # no raw_scores in the request -> unchanged raw behaviour
    for i, x in r.items():
        assert 0.0 < s[i] < 1.0
        assert abs(s[i] - 1.0 / (1.0 + math.exp(-x))) < 1e-6
    # the transform is monotone, so the ranking is unchanged
    assert [e["index"] for e in sig.body["results"]] == [e["index"] for e in raw.body["results"]]


@pytest.mark.parametrize("top_n", [0, -1])
def test_rerank_nonpositive_top_n_returns_all(top_n):
    global server
    server.start()
    res = server.make_request("POST", "/rerank", data={
        "query": "Machine learning is", "documents": TEST_DOCUMENTS, "top_n": top_n})
    assert res.status_code == 200
    assert len(res.body["results"]) == len(TEST_DOCUMENTS)


def test_rerank_return_documents():
    global server
    server.start()
    base = {"query": "Machine learning is", "documents": TEST_DOCUMENTS}
    on = server.make_request("POST", "/rerank", data=dict(base, return_documents=True))
    off = server.make_request("POST", "/rerank", data=base)
    assert on.status_code == off.status_code == 200
    for e in on.body["results"]:
        assert e["document"] == {"text": TEST_DOCUMENTS[e["index"]]}
    for e in off.body["results"]:
        assert "document" not in e
