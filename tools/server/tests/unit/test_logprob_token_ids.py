# logprob_token_ids request field (rows S69-VLLM-PICK-LOGPROB-TOKEN-IDS-2026-09-29-01 and
# S69-aphrodite-PICK-LOGPROB-FOR-SPECIFIC-TOKEN-IDS-2026-09-29-01).
#
# SOURCE-READ: engines/vllm/vllm/sampling_params.py:297-309 (the field: at most 128 ids, all inside the vocabulary). Apache-2.0;
# reimplemented against this server's per-token readout, no code copied.
#
# Oracle: the existing n_probs path over the whole vocabulary (stories260K, n_vocab 512). It ranks every token and reports
# log(prob) of each; the new field must report the same number for the ids asked for, including ids far below the top.
import math
import pytest

from utils import ServerPreset, use_local_stories260k  # noqa: E402

server = ServerPreset.tinyllama2()

N_VOCAB = 512
PROMPT = "Once upon a time there was a little girl named Lily who loved to"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 1


def _gen(srv, **extra):
    data = {"prompt": PROMPT, "n_predict": 1, "temperature": 0.0, "seed": 42, "cache_prompt": False}
    data.update(extra)
    return srv.make_request("POST", "/completion", data=data)


def _full_distribution(srv):
    """{token id: logprob} for the first generated position, from the full n_probs list."""
    res = _gen(srv, n_probs=N_VOCAB, post_sampling_probs=False)
    assert res.status_code == 200
    top = res.body["completion_probabilities"][0]["top_logprobs"]
    assert len(top) == N_VOCAB, "control: the oracle must list every token"
    return {e["id"]: e["logprob"] for e in top}, top


def test_logprob_token_ids_match_the_full_distribution():
    global server
    server.start()
    dist, top = _full_distribution(server)
    best = top[0]["id"]
    worst = top[-1]["id"]
    middle = top[N_VOCAB // 2]["id"]
    ids = [worst, best, middle]
    assert len(set(ids)) == 3

    res = _gen(server, logprob_token_ids=ids)
    assert res.status_code == 200
    entry = res.body["completion_probabilities"][0]
    got = {e["id"]: e["logprob"] for e in entry["token_id_logprobs"]}
    assert set(got) == set(ids)
    for tok in ids:
        assert abs(got[tok] - dist[tok]) < 1e-4, f"token {tok}: got {got[tok]}, oracle {dist[tok]}"
    assert got[worst] < got[middle] < got[best], "control: the three ids sit at very different ranks"
    # the asked-for id far below the top is NOT in the default top list (n_probs was switched on at 1)
    assert [e["id"] for e in entry["top_logprobs"]] == [best]
    assert sum(math.exp(v) for v in got.values()) <= 1.0 + 1e-4


def test_logprob_token_ids_same_under_post_sampling_probs():
    global server
    server.start()
    dist, top = _full_distribution(server)
    ids = [top[-1]["id"], top[3]["id"]]
    res = _gen(server, logprob_token_ids=ids, post_sampling_probs=True)
    assert res.status_code == 200
    got = {e["id"]: e["logprob"] for e in res.body["completion_probabilities"][0]["token_id_logprobs"]}
    for tok in ids:
        assert abs(got[tok] - dist[tok]) < 1e-4


def test_logprob_token_ids_reported_at_every_position():
    global server
    server.start()
    res = _gen(server, n_predict=4, logprob_token_ids=[5, 6])
    assert res.status_code == 200
    probs = res.body["completion_probabilities"]
    assert len(probs) == 4
    for p in probs:
        assert [e["id"] for e in p["token_id_logprobs"]] == [5, 6]
        assert all(e["logprob"] <= 0.0 for e in p["token_id_logprobs"])
    # the positions differ, so the readout is per position and not copied from the first
    assert probs[0]["token_id_logprobs"] != probs[1]["token_id_logprobs"]


def test_logprob_token_ids_absent_unless_asked():
    global server
    server.start()
    res = _gen(server, n_probs=2)
    assert res.status_code == 200
    assert "token_id_logprobs" not in res.body["completion_probabilities"][0]


def test_logprob_token_ids_duplicates_are_folded():
    global server
    server.start()
    res = _gen(server, logprob_token_ids=[7, 7, 8])
    assert res.status_code == 200
    assert [e["id"] for e in res.body["completion_probabilities"][0]["token_id_logprobs"]] == [7, 8]


@pytest.mark.parametrize("bad", [[N_VOCAB], [-1], ["a"], [1.5], "5", list(range(129))])
def test_logprob_token_ids_invalid_is_refused(bad):
    global server
    server.start()
    res = _gen(server, logprob_token_ids=bad)
    assert res.status_code == 400
    assert "logprob_token_ids" in str(res.body)


def test_logprob_token_ids_exactly_128_is_accepted():
    global server
    server.start()
    res = _gen(server, logprob_token_ids=list(range(128)))
    assert res.status_code == 200
    assert len(res.body["completion_probabilities"][0]["token_id_logprobs"]) == 128
