"""Strict JSON-schema check, row S69-SGLANG-PICK-STRICT-JSON-SCHEMA-CHECK-2026-10-18-01.

json_schema_to_grammar() silently drops keywords it cannot express (multipleOf, uniqueItems, not, ...).
With "json_schema_strict": true (or the server flag --json-schema-strict) such a schema is refused with 400 and
a JSON pointer per dropped constraint, the way SGLang refuses per backend
(sglang python/sglang/srt/constrained/json_schema_validation.py, Apache-2.0). Off by default: the old behaviour
(accept, constrain what can be constrained) is kept for existing clients.
"""
import pytest
from utils import ServerPreset

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _req(schema, **extra):
    data = {"prompt": "Number:", "n_predict": 4, "json_schema": schema}
    data.update(extra)
    return server.make_request("POST", "/completion", data=data)


def test_strict_refuses_a_dropped_keyword_with_its_pointer():
    server.start()
    res = _req({"type": "object", "properties": {"n": {"type": "integer", "multipleOf": 7}}}, json_schema_strict=True)
    assert res.status_code == 400, res.body
    msg = res.body["error"]["message"]
    assert "#/properties/n" in msg and "multipleOf" in msg


def test_strict_accepts_a_fully_enforced_schema():
    server.start()
    res = _req({"type": "integer", "minimum": 0, "maximum": 99}, json_schema_strict=True)
    assert res.status_code == 200, res.body


def test_default_is_lenient():
    server.start()
    res = _req({"type": "integer", "multipleOf": 7})
    assert res.status_code == 200, res.body
    res = _req({"type": "integer", "multipleOf": 7}, json_schema_strict=False)
    assert res.status_code == 200, res.body


def test_server_flag_sets_the_default():
    server.extra_args = ["--json-schema-strict"]
    server.start()
    res = _req({"type": "string", "format": "email"})
    assert res.status_code == 400, res.body
    assert "format" in res.body["error"]["message"]
    # a request can still opt out
    res = _req({"type": "string", "format": "email"}, json_schema_strict=False)
    assert res.status_code == 200, res.body
