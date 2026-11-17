import pytest
from utils import *
from utils import use_local_stories260k  # explicit, so static checkers see where it comes from

# koboldcpp grammar_retain_state port, adopted per-request (gpttype_adapter.cpp:6035-6046,
# SOURCE-READ, AGPL-3.0, no code copied): a USER GBNF grammar is normally rebuilt from its root on
# every request (common/sampling.cpp -- common_grammar_needs_prefill() deliberately excludes
# COMMON_GRAMMAR_TYPE_USER from the automatic generation_prompt replay). grammar_continue_prefix is
# the explicit, per-request opt-in that replays caller-supplied already-generated text through a
# fresh grammar sampler, landing at the same parse state a "retained" grammar would have had.
#
# root ::= "AAAA" forces the model to emit exactly four 'A' characters, then only EOS is grammar-
# valid. With grammar_continue_prefix="AA" the sampler starts already 2 characters into the literal,
# so it must force exactly the remaining "AA" -- proving it did NOT restart at the grammar's root
# (which would force "AAAA" again). The companion test is the negative control: no prefix -> root.

GRAMMAR = 'root ::= "AAAA"'

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_port = 8091
    # Same TLS-less build_cpu limitation as test_tool_call_fix.py (ROOT-CAUSE/CONFIGS-TRIED/
    # RESEARCH-DONE documented there and in the findings file): point at a local model file
    # instead of letting the server binary try its own --hf-repo HTTPS fetch.
    # Resolved from the model cache, not the gitignored tests/.test_models (see utils.local_model_path).
    use_local_stories260k(server)


def test_grammar_without_continue_prefix_restarts_at_root():
    """Negative control: a USER grammar with no continuation prefix forces the FULL literal."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 16,
        "prompt": "Once upon a time",
        "grammar": GRAMMAR,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    assert res.body["content"] == "AAAA"
    assert res.body["generation_settings"]["grammar_continue_prefix"] == ""


def test_grammar_continue_prefix_continues_user_grammar():
    """Positive case: grammar_continue_prefix='AA' must force only the remaining 'AA', not a restart."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 16,
        "prompt": "Once upon a time",
        "grammar": GRAMMAR,
        "grammar_continue_prefix": "AA",
        "temperature": 0.0,
    })
    assert res.status_code == 200
    assert res.body["content"] == "AA"
    assert res.body["generation_settings"]["grammar_continue_prefix"] == "AA"
