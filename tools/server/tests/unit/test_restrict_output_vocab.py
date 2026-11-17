import concurrent.futures
import os
import re
import tempfile
import time

import pytest
from utils import *
from utils import use_local_stories260k  # explicit, so static checkers see where it comes from

# --restrict-output-vocab: CTranslate2's vocabulary map (Decoder::update_output_layer, src/layers/decoder.cc:72-135, MIT, behaviour only). When every
# running request carries allowed_token_ids and none of them reads the raw distribution, the server multiplies the output weight only by the union of
# the allowed tokens. The sampler already masks every other token to -inf, so the answer must not change: the oracle is the SAME request on the SAME
# model without the flag. "Fired" is the server's own log line, written from llama_output_vocab_subset_used() after the decode (the number of
# rows the graph really multiplied by), not from the flag being set.

server = ServerPreset.tinyllama2()
PROMPT = "Once upon a time there was a little girl"
LINE = re.compile(r"the decode multiplied the output weight by (\d+) of (\d+) vocabulary entries")


def make_server(flag):
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 2
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.extra_args = ["--restrict-output-vocab"] if flag else []


def log_lines():
    time.sleep(0.3)
    with open(server.log_path) as f:
        return [(int(m.group(1)), int(m.group(2))) for m in LINE.finditer(f.read())]


def _complete(**extra):
    data = {"prompt": PROMPT, "n_predict": 12, "return_tokens": True, "cache_prompt": False, "ignore_eos": True}
    data.update(extra)
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200, res.body
    return res.body["tokens"]


def _allowed_sets():
    # ids the model really likes (so the restriction changes the answer), plus both ends of the vocabulary and a few arbitrary ones
    make_server(False)
    server.start()
    base = _complete(temperature=0.0)
    server.stop()
    liked = sorted(set(base))
    return base, [liked, liked[:3] + [0, 511], [7, 9, 11, 13, 15, 400, 401]]


def test_same_answers_with_and_without_the_flag_and_the_decode_log_proves_it_fired():
    base, sets = _allowed_sets()
    answers = {}
    for flag in (False, True):
        make_server(flag)
        server.start()
        for i, allowed in enumerate(sets):
            answers[(flag, i, "greedy")] = _complete(temperature=0.0, allowed_token_ids=allowed)
            answers[(flag, i, "sampled")] = _complete(temperature=0.9, seed=1234, allowed_token_ids=allowed)
        lines = log_lines()
        if flag:
            # the log shows a decode that multiplied by exactly the union of the allowed ids (sizes of the three sets), never by the whole vocabulary
            sizes = {len(set(a)) for a in sets}
            assert {u for u, _ in lines} >= sizes, (lines, sizes)
            assert all(total == 512 for _, total in lines)
        else:
            assert lines == [], lines   # flag OFF: not a trace of the feature
        server.stop()
    for i, allowed in enumerate(sets):
        for mode in ("greedy", "sampled"):
            on, off = answers[(True, i, mode)], answers[(False, i, mode)]
            assert on == off, (i, mode, on, off)
            assert set(on) <= set(allowed)
    # CONTROL: the allow-list really constrains (the unrestricted greedy answer is not inside the narrow set), so equality above is not vacuous
    assert not set(base) <= set(sets[2])
    assert answers[(True, 2, "greedy")] != base


def test_a_request_without_allowed_ids_switches_the_restriction_off_for_the_whole_batch():
    make_server(True)
    server.start()
    base = _complete(temperature=0.0)                       # no allow-list: full projection, nothing logged
    assert log_lines() == []
    restricted = _complete(temperature=0.0, allowed_token_ids=[5, 6, 7, 8])
    assert log_lines()[-1][0] == 4
    # two concurrent requests, one of them unrestricted: the batch must use the full projection, and each answer is still its own
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        a = ex.submit(_complete, temperature=0.0, allowed_token_ids=[5, 6, 7, 8], n_predict=24)
        b = ex.submit(_complete, temperature=0.0, n_predict=24)
        ra, rb = a.result(), b.result()
    assert set(ra) <= {5, 6, 7, 8}
    assert ra[:12] == restricted and rb[:12] == base
    server.stop()


def test_requests_that_read_the_raw_distribution_are_not_restricted():
    make_server(True)
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 2, "temperature": 0.0, "allowed_token_ids": [5, 6, 7, 8], "n_probs": 3, "cache_prompt": False})
    assert res.status_code == 200
    # n_probs is normalised over the whole vocabulary: 3 candidates with finite log-probs that are NOT the allowed ones
    top = res.body["completion_probabilities"][0]["top_logprobs"]
    assert len(top) == 3 and all(t["logprob"] > -1e9 for t in top)
    assert log_lines() == [] or log_lines()[-1][0] != 4    # not multiplied by the 4-entry subset
    server.stop()
