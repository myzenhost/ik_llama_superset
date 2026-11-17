"""Expiring logit bias (ik_llama.cpp #1731/#1914) through tools/server: the "expiring_logit_bias" request field and
--expiring-logit-bias-file, the sampler text feed they depend on, and their interplay with banned strings and drafting.

The sampler (common/sampling.cpp) decides when a rule applies from the text generated so far (common_sampler::generated_text
+ playing_text) and counts generated tokens (is_decoding). tools/server filled none of that, so every rule was inert; the
merge from ik main kept the sampler half and left the server half out (row
S69-IKMERGE-TOOLS-SERVER-LACKS-REWIND-BANPHRASE-ELB-WIRING-2026-09-30-01).

Rule format: (DURATION : "phrase" : BIAS) biases the phrase's tokens for DURATION generated tokens; the first token is
biased while fewer than DURATION tokens have been generated, each later token only when the text so far ends with the
phrase's earlier tokens. Model: stories260K, greedy, so every output is reproducible; each test compares with a control.

stories260K tokenizes "zqx" as [' ', 'z', 'q', 'x'] and 'y' / 'a' as the single tokens ' y' / ' a' (measured 2026-10-01).
"""
import glob
import os

import pytest

from utils import ServerPreset

server = ServerPreset.tinyllama2()

PROMPT = "Once upon a time"
N_PREDICT = 24
FORCED = " zqx"          # the text a (8 : "zqx" : 50) rule forces at the start; the greedy reference never produces it


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_port = 18434      # NOT 8081: that is the live serving instance on this host


def _complete(**extra) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": N_PREDICT, "temperature": 0, "cache_prompt": False, **extra,
    })
    assert res.status_code == 200, res.body
    return res.body


def _ref() -> str:
    return _complete()["content"]


# one bias per token of the phrase [' ', 'z', 'q', 'x']. A single bias would be extended over all four, and then the first
# token ' ' (which stays biased for the whole duration, as its bias is not tied to the text) ties with 'q' after ' z' and the
# output becomes ' z z z zqx' (measured 2026-10-01); rising biases make the phrase come out whole.
FORCE_RULE = ['(8 : "zqx" : 50, 60, 70, 80)']


def test_rule_changes_the_output_and_forces_the_phrase():
    server.start()
    ref = _ref()
    assert FORCED not in ref
    got = _complete(expiring_logit_bias=FORCE_RULE)
    assert got["content"].startswith(FORCED), got["content"]
    assert got["content"] != ref
    assert got["generation_settings"]["expiring_logit_bias_stages"] >= 1


def test_without_the_field_nothing_changes():
    # control: no field, an empty list, and a comment-only list all leave the greedy output byte-identical
    server.start()
    ref = _ref()
    assert _complete(expiring_logit_bias=[])["content"] == ref
    got = _complete(expiring_logit_bias=["# nothing here", ""])
    assert got["content"] == ref
    assert got["generation_settings"]["expiring_logit_bias_stages"] == 0
    assert _complete()["generation_settings"]["expiring_logit_bias_stages"] == 0


def test_a_rule_does_not_leak_into_the_next_request():
    server.start()
    ref = _ref()
    assert _complete(expiring_logit_bias=FORCE_RULE)["content"].startswith(FORCED)
    assert _complete()["content"] == ref


def test_bias_expires_after_its_token_count():
    # greedy output is ", there was ..."; " there" is generated at index 1. A -60 bias on it for 2 tokens still covers index 1
    # (generated count 1 < 2) and changes the text there; for 1 token it has expired by then and the output is the reference.
    server.start()
    ref = _ref()
    assert ref.startswith(", there"), ref
    expired = _complete(expiring_logit_bias=['(1 : "there" : -60)'])["content"]
    active = _complete(expiring_logit_bias=['(2 : "there" : -60)'])["content"]
    assert expired == ref, "a 1-token bias must be gone by the second token"
    assert active != ref and not active.startswith(", there"), active


def test_persistent_rule_does_not_expire():
    # ((...)) without a duration never expires: " there" never comes back; the reference has it, an expiring rule allows it later
    server.start()
    ref = _ref()
    assert " there" in ref
    persistent = _complete(expiring_logit_bias=['(("there" : -60))'], n_predict=48)["content"]
    assert " there" not in persistent, persistent


def test_bad_rule_is_rejected():
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 4, "expiring_logit_bias": ['(x : "zq" : 5)']})
    assert res.status_code == 400, res.body
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 4, "expiring_logit_bias": "not a list"})
    assert res.status_code == 400, res.body


def test_flag_file_applies_the_same_rule(tmp_path):
    f = tmp_path / "elb.txt"
    f.write_text("# forces ' zqx' at the start\n" + FORCE_RULE[0] + "\n", encoding="utf-8")
    server.extra_args = ["--expiring-logit-bias-file", str(f)]
    server.start()
    assert _complete()["content"].startswith(FORCED)


def test_rule_sees_text_held_back_by_banned_strings():
    # With a banned string the last tokens wait in the hold-back buffer and are NOT in the slot's text yet. The bias for
    # 'z', 'q', 'x' depends on the text so far ending with the earlier tokens, so it only works if the sampler is also fed
    # the held tokens (playing_text). The banned string never occurs, so nothing is rewound: only the feed is tested.
    server.start()
    plain = _complete(expiring_logit_bias=FORCE_RULE)["content"]
    held = _complete(expiring_logit_bias=FORCE_RULE, banned_strings=["qqqqqqqqqqqq"])["content"]
    assert plain.startswith(FORCED)
    assert held == plain


def test_rewind_restores_the_bias_schedule():
    # two single-token rules for the first position: ' y' (+50) beats ' a' (+40). The reference starts ' y'. Banning ' y' rewinds
    # to position 0 and bans that token there; the schedule must be back at 'zero tokens generated', so ' a' (+40) is still
    # live and wins. A schedule that kept the count of the discarded token would have expired both rules (duration 1).
    server.start()
    rules = ['(1 : "y" : 50)', '(1 : "a" : 40)']
    free = _complete(expiring_logit_bias=rules)["content"]
    assert free.startswith(" y"), free
    got = _complete(expiring_logit_bias=rules, banned_strings=[" y"])["content"]
    assert got.startswith(" a"), got
    # control: without the rules the ban alone gives the natural continuation, not ' a'
    assert not _complete(banned_strings=[" y"])["content"].startswith(" a")


def _draft_path() -> str:
    for r in (os.environ.get("LLAMA_CACHE", ""), os.environ.get("HUGGINGFACE_HUB_CACHE", "")):
        if r:
            hits = glob.glob(os.path.join(r, "models--ggml-org--test-model-stories260K", "snapshots", "*", "stories260K-f32.gguf"))
            if hits:
                return hits[0]
    pytest.skip("stories260K-f32.gguf not in the HF cache (LLAMA_CACHE / HUGGINGFACE_HUB_CACHE)")


@pytest.mark.parametrize("draft_sample", [False, True])
def test_condition_on_two_accepted_pieces_inside_one_verification_run(draft_sample):
    # The reference starts ", there was a little girl". The rule suppresses ' a' only when the text so far ends with
    # ' there was' (two tokens). The self-drafting model proposes the whole reference, so ' there' and ' was' are accepted
    # in ONE verification run and ' a' is judged against text the sampler assembled itself from those accepted pieces
    # (common_sampler_accept). If that text holds each piece twice (' there there was was') the condition never matches and
    # ' a' slips through; the output must equal the run without drafting.
    global server
    rule = ['(("there was a" : 0, 0, -60))']
    server.start()
    ref = _ref()
    assert " was a" in ref, ref
    plain = _complete(expiring_logit_bias=rule)["content"]
    assert " was a" not in plain, plain
    server.stop()

    server = ServerPreset.tinyllama2()
    server.server_port = 18434
    server.n_ctx = 2048
    server.fa = "off"
    server.model_draft = _draft_path()
    server.spec_type = "draft-simple"
    server.spec_draft_n_max = 8
    server.extra_args = ["--spec-draft-p-min", "0"] + (["--spec-draft-sample"] if draft_sample else [])
    server.start()
    got = _complete(expiring_logit_bias=rule)
    assert got["timings"].get("draft_n", 0) > 0, "no draft was proposed; the speculative path was not exercised"
    assert got["timings"].get("draft_n_accepted", 0) > 0, "no draft token was accepted, so no run held two accepted pieces"
    assert got["content"] == plain


@pytest.mark.parametrize("draft_sample", [False, True])
def test_rule_works_with_speculative_decoding(draft_sample):
    # The same model drafts for itself; at temperature 0 the output must equal the run without drafting. The rule's
    # later tokens only apply if the sampler sees the tokens accepted earlier in the SAME verification run.
    # With --spec-draft-sample the verifier is the rejection-sampling variant, which used to append every piece to the
    # sampler's text twice (so ' zqx' could not be completed).
    global server
    server.start()
    plain = _complete(expiring_logit_bias=FORCE_RULE)["content"]
    server.stop()

    server = ServerPreset.tinyllama2()
    server.server_port = 18434
    server.n_ctx = 2048
    server.fa = "off"
    server.model_draft = _draft_path()
    server.spec_type = "draft-simple"
    server.spec_draft_n_max = 8
    server.extra_args = ["--spec-draft-p-min", "0"] + (["--spec-draft-sample"] if draft_sample else [])
    server.start()
    got = _complete(expiring_logit_bias=FORCE_RULE)
    assert got["timings"].get("draft_n", 0) > 0, "no draft was proposed; the speculative path was not exercised"
    assert got["content"] == plain
    assert got["content"].startswith(FORCED)
