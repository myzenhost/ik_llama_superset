# Grammar-validated drafts: with a grammar active, every draft is cut to its grammar-valid prefix BEFORE the verify batch, whatever
# proposer made it (row S69-VLLM-W2-GRAMMAR-VALIDATED-DRAFTS-2026-10-27-01).
# vLLM (Apache-2.0, sha 924707f1bf94) vllm/v1/structured_output/__init__.py:294-312 validate_tokens, called from
# vllm/v1/core/sched/scheduler.py:2487 and :2515, backend_guidance.py:185-200; read, rewritten here as a walk over a copy of the
# slot's grammar, no code copied. The fork's --spec-draft-grammar only constrains a DRAFT MODEL; an n-gram / corpus / suffix drafter
# proposes from history and had no check at all, so its grammar-forbidden tokens filled the verify batch.
#
# Setup: random-weight tiny model, n-gram drafter (copies history). The prompt holds "aaa" followed by "bbb...", the grammar allows
# only 'a', so the drafter's continuation of "aa" is [a, b, b, ...]: one valid token, the rest forbidden.
#  - validation OFF (--no-spec-draft-validate): the forbidden tail is drafted and verified, then rejected (accepted < drafted).
#  - validation ON  (default): only the valid prefix is drafted; every drafted token is accepted.
#  - the generated text is identical either way (lossless: the target decides every token).
#  - control, no grammar: the validator is inert, drafted counts are identical.
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tiny_random_model import tiny_server  # noqa: E402

A, B = 4, 5     # token ids of "a" and "b" in the tiny vocabulary (tiny_random_model.py: pad, s, /s, newline, then a-z)
PROMPT = [1] + [A] * 3 + [B] * 12


def run(tmp_path, validate, grammar, prompt=None):
    os.makedirs(str(tmp_path), exist_ok=True)
    srv = tiny_server(tmp_path)
    srv.spec_type = "ngram-simple"
    srv.spec_draft_n_min = 1
    srv.spec_draft_n_max = 8
    srv.n_predict = 24
    srv.extra_args = ["--spec-ngram-simple-size-n", "2", "--spec-ngram-simple-size-m", "8"]
    if not validate:
        srv.extra_args.append("--no-spec-draft-validate")
    srv.start()
    try:
        body = {"prompt": prompt or PROMPT, "n_predict": 24, "temperature": 0.0, "return_tokens": True, "seed": 1}
        if grammar:
            body["grammar"] = grammar if isinstance(grammar, str) else "root ::= [a]{40}"
        res = srv.make_request("POST", "/completion", data=body)
        assert res.status_code == 200, res.body
    finally:
        srv.stop()
    t = res.body["timings"]
    print("validate=%s grammar=%s draft_n=%d accepted=%d tokens=%s" % (validate, bool(grammar), t.get("draft_n", 0), t.get("draft_n_accepted", 0),
                                                                      res.body["tokens"]))
    return res.body["tokens"], t.get("draft_n", 0), t.get("draft_n_accepted", 0)


def test_grammar_forbidden_draft_tail_is_not_verified(tmp_path):
    tok_off, n_off, acc_off = run(tmp_path / "off", False, True)
    tok_on, n_on, acc_on = run(tmp_path / "on", True, True)
    assert tok_on == tok_off                       # lossless
    assert set(tok_on) == {A}                      # the grammar really constrained the output
    assert n_off > 0, "the n-gram drafter proposed nothing: the test has no teeth"
    assert acc_off < n_off, "without validation the forbidden tail must have been drafted and rejected"
    assert n_on < n_off                            # the forbidden tokens were not sent to verify
    assert acc_on == n_on                          # every token that still goes to verify is accepted


def test_no_grammar_validator_is_inert(tmp_path):
    tok_off, n_off, acc_off = run(tmp_path / "off", False, False)
    tok_on, n_on, acc_on = run(tmp_path / "on", True, False)
    assert tok_on == tok_off
    assert (n_on, acc_on) == (n_off, acc_off)


def test_valid_prefix_of_the_draft_is_kept_and_only_the_cut_tail_was_wasted(tmp_path):
    # ngram-simple copies the continuation of the MOST RECENT earlier match of (last token, sampled token) and needs the history to be
    # longer than size_n + size_m + 1 (common/ngram-map.cpp:49-111). History = S S with S = a a a a b b b b b b, so the draft is a
    # run of a/b copied from S. The grammar is "aaaabb" then 30 x 'a'. A draft token that the grammar forbids can never be accepted; a
    # draft token the grammar allows may still lose to the target (grammar-valid is not target-chosen), so the claim is NOT
    # "everything kept is accepted" but:
    #  - the output is the same;
    #  - a non-empty valid prefix survives (0 < n_on < n_off): the cut is at the first forbidden token, not the whole draft;
    #  - the number accepted is unchanged (acc_on == acc_off): nothing the target would have taken was cut;
    #  - the verify batch wasted fewer tokens (n_on - acc_on < n_off - acc_off).
    prompt = [1] + ([A] * 4 + [B] * 6) * 2
    grammar = 'root ::= "aaaabb" [a]{30}'
    tok_off, n_off, acc_off = run(tmp_path / "off", False, grammar, prompt)
    tok_on, n_on, acc_on = run(tmp_path / "on", True, grammar, prompt)
    assert tok_on == tok_off
    assert n_off > 0 and acc_off < n_off, (n_off, acc_off)
    assert 0 < n_on < n_off, (n_on, n_off)
    assert acc_on == acc_off, (acc_on, acc_off)
    assert (n_on - acc_on) < (n_off - acc_off), (n_on, acc_on, n_off, acc_off)
