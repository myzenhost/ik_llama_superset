"""top_p with top_k off on served input, row S69-MLC-LLM-PICK-TOPK-TOPP-WITHOUT-FULL-SORT-2026-10-18-01.

With top_k 0 the queue defers its sort and top_p runs on unsorted input (llama_sample_top_p_unsorted in
src/llama-sampling.cpp). The served nucleus must be the smallest probability-sorted prefix of the full
distribution whose mass reaches top_p, renormalised. The full distribution comes from the same request
with top_p 1.0. Both the old (sort-everything) and the new path must pass; this proves the new path on a
live server, not a speed difference.
"""
import pytest
from utils import ServerPreset

server = ServerPreset.tinyllama2()

PROMPTS = ["Once upon a time, there was a little girl who",
           "The cat sat on the",
           "Tom went to the park and"]
N_PROBS = 600  # above the stories260K vocabulary (512), so every candidate is reported


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _probs(prompt: str, top_p: float) -> list:
    data = {"prompt": prompt, "n_predict": 1, "n_probs": N_PROBS, "post_sampling_probs": True,
            "temperature": 1.0, "top_k": 0, "top_p": top_p, "min_p": 0.0, "seed": 42, "cache_prompt": False}
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200, res.body
    return [(t["id"], t["prob"]) for t in res.body["completion_probabilities"][0]["top_probs"]]


@pytest.mark.parametrize("top_p", [0.5, 0.9, 0.95])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_nucleus_matches_the_full_distribution(prompt, top_p):
    server.start()
    full = sorted(_probs(prompt, 1.0), key=lambda x: (-x[1], x[0]))
    assert len(full) > 100, "premise: the whole vocabulary is reported"
    cum, cut = 0.0, 0
    for i, (_, p) in enumerate(full):
        cum += p
        if cum >= top_p:
            cut = i + 1
            break
    margin = abs(cum - top_p)
    got = dict(_probs(prompt, top_p))
    if margin > 1e-4:  # boundary not within rounding of top_p: the set is determined
        assert set(got) == {t for t, _ in full[:cut]}, (len(got), cut)
    mass = sum(p for _, p in full[:cut])
    for tid, p in full[:cut]:
        if tid in got:
            assert got[tid] == pytest.approx(p / mass, rel=2e-3, abs=1e-5), tid
