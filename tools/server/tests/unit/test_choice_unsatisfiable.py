# A choice/grammar the vocabulary cannot satisfy fails THAT request with 400 and leaves the server serving.
#
# The tiny model's vocabulary is lower-case a-z, '"', newline and filler, so "ZZZ" cannot be spelled: the grammar masks
# every candidate to -inf. Before the fix the sampler hit GGML_ABORT (llama-sampling.cpp, "Failed to sample token"),
# wrote probabilities.txt into the server's CWD and the whole process died, so the follow-up request could not connect.
# Row S69-FORK-UNSATISFIABLE-GRAMMAR-ABORTS-SERVER-SAMPLING-839-2026-10-11-01.
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tiny_random_model import PROMPT, tiny_server  # noqa: E402


def _gen(srv, **extra):
    body = {"prompt": PROMPT, "n_predict": 8, "temperature": 1.0, "seed": 7}
    body.update(extra)
    return srv.make_request("POST", "/completion", data=body)


@pytest.mark.parametrize("constraint", [
    {"choice": ["ZZZ"]},
    {"grammar": 'root ::= "ZZZ"'},
    {"choice": ["ZZZ"], "temperature": 0.0},     # greedy used to return a token the grammar does not allow
])
def test_unsatisfiable_constraint_is_a_request_error(tmp_path, capfd, constraint):
    dump = os.path.join(os.getcwd(), "probabilities.txt")     # the server inherits the test's CWD
    dump_before = os.path.getmtime(dump) if os.path.exists(dump) else None
    srv = tiny_server(tmp_path)
    srv.start()
    try:
        r = _gen(srv, **constraint)
        assert r.status_code == 400, r.body
        assert "cannot be satisfied" in r.body["error"]["message"], r.body
        # the same server answers the next request: it did not abort
        ok = _gen(srv, choice=["yes", "no"])
        assert ok.status_code == 200, ok.body
        assert ok.body["content"] in ("yes", "no"), ok.body
    finally:
        srv.stop()
    out = capfd.readouterr()
    assert "Failed to sample token" not in out.out + out.err
    dump_after = os.path.getmtime(dump) if os.path.exists(dump) else None
    assert dump_after == dump_before, "the sampler wrote probabilities.txt into the server CWD"
