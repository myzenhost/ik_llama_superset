# `choice`: output is exactly one of the listed strings (vLLM structured_outputs.choice,
# vllm/v1/structured_output/utils.py:438-455). Random-weight model: without the constraint the output is not in the set.
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tiny_random_model import PROMPT, tiny_server  # noqa: E402

CHOICES = ["yes", 'n"o', "may\nbe"]


@pytest.fixture
def srv(tmp_path):
    server = tiny_server(tmp_path)
    server.start()
    yield server
    server.stop()


def _gen(srv, **extra):
    body = {"prompt": PROMPT, "n_predict": 16, "temperature": 1.0}
    body.update(extra)
    return srv.make_request("POST", "/completion", data=body)


def test_choice_constrains_output(tmp_path, capfd):
    srv = tiny_server(tmp_path)
    srv.start()
    try:
        for seed in range(1, 7):
            r = _gen(srv, seed=seed, choice=CHOICES)
            assert r.status_code == 200, r.body
            assert r.body["content"] in CHOICES, r.body["content"]
        unconstrained = {_gen(srv, seed=s).body["content"] for s in range(1, 4)}
        assert not (unconstrained <= set(CHOICES))                                  # control: the constraint did the work
    finally:
        srv.stop()
    assert capfd.readouterr().out.count("choice constraint: 3 alternatives") >= 6   # FIRED


@pytest.mark.parametrize("bad", [[], [""], "yes", [1]])
def test_choice_rejects_bad_values(srv, bad):
    assert _gen(srv, choice=bad).status_code == 400


def test_choice_excludes_other_constraints(srv):
    assert _gen(srv, choice=["a"], regex="a").status_code == 400
    assert _gen(srv, choice=["a"], grammar='root ::= "a"').status_code == 400
