# --slot-evict-lookahead N: the LRU branch of get_available_slot keeps an idle slot that a waiting request would reuse
# (row S69-NANOVLLM-PICK-QUEUE-AWARE-SLOT-EVICTION-2026-10-18-01; idea from nano-vLLM PR #278, MIT, description read,
# no code copied).
#
# Two slots: slot 0 holds prefix A (used first, so it is the least recently used), slot 1 holds prefix B. One request
# then carries two prompts, posted to the queue together: U (unlike A and B, so it takes a slot by LRU) and A + more.
# Plain LRU gives U slot 0 and evicts A, so A + more lands on slot 1 and prefills A again. With the lookahead U sees
# A + more waiting, keeps slot 0 and takes slot 1; A + more then reuses slot 0's cached prefix.
import pytest
from utils import ServerPreset

PREFIX_A = "Once upon a time there was a little girl named Lily who loved to play outside in the park with her friends. " * 3
PREFIX_B = "The big brown dog ran very fast across the green field to catch the red ball that Tom threw. " * 3
PROMPT_U = "Sam had a toy car and a small blue boat, and he wanted to sail it on the pond near his house today."


@pytest.fixture
def evict_server():
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.cache_ram = 0                      # no RAM prompt cache: an evicted prefix is gone, not reloaded
    server.no_cache_publish_on_prefill = True  # no cross-slot sharing: the measured reuse is the slot's own cache
    return server


def tokenize(server, text):
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    assert res.status_code == 200
    return res.body["tokens"]


def run_scenario(server):
    server.start()
    for id_slot, text in [(0, PREFIX_A), (1, PREFIX_B)]:
        res = server.make_request("POST", "/completion", data={"prompt": text, "id_slot": id_slot, "cache_prompt": True})
        assert res.status_code == 200
    n_a = len(tokenize(server, PREFIX_A))
    res = server.make_request("POST", "/completion", data={
        "prompt": [PROMPT_U, PREFIX_A + "She saw a bird in the tree."],
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert isinstance(res.body, list) and len(res.body) == 2
    u, a = res.body
    print("U:", u["id_slot"], u["timings"], " A+more:", a["id_slot"], a["timings"], " len(A) =", n_a)
    return u, a, n_a


# control: plain LRU (the default) evicts the prefix the waiting request needed
def test_plain_lru_loses_the_waiting_prefix(evict_server):
    u, a, n_a = run_scenario(evict_server)
    assert u["id_slot"] == 0
    assert a["id_slot"] == 1
    assert a["timings"]["cache_n"] < n_a // 2


def test_lookahead_keeps_the_waiting_prefix(evict_server):
    evict_server.extra_args = ["--slot-evict-lookahead", "16"]
    u, a, n_a = run_scenario(evict_server)
    assert u["id_slot"] == 1
    assert a["id_slot"] == 0
    assert a["timings"]["cache_n"] >= n_a - 1
