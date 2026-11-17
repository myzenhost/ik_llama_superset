# --spec-draft-grammar: the draft model drafts under the request's grammar, so it does not propose tokens the grammar
# forbids (row S69-LMDEPLOY-PICK-GRAMMAR-MASKED-SPEC-DRAFT-2026-10-18-01). LMDeploy v0.17.0 (Apache-2.0) masks each draft
# step with a forked grammar matcher (lmdeploy/pytorch/spec_decode/guided_spec_helper.py:69-98, :124-167, rollback
# :221-231); read, rewritten here as a grammar copy in the draft sampler, no code copied.
#
# Target stories15M MoE (F16), greedy, a grammar that allows only the letters a-e and spaces. Without the flag the
# draft proposes ordinary story words and the target refuses most of them; with it the draft proposes the grammar's own
# argmax. The output must not change (lossless) in either test.
#  - test 1, draft = stories15M q4_0 (a DIFFERENT model): the two models still disagree on which legal token is best, so
#    acceptance only rises, it does not reach 1. MEASURED 2026-10-19 (roster4 fix2): off 17/224 = 0.076, on 34/96 = 0.354.
#  - test 2, draft = the target's own file: every disagreement left is the grammar's, so with the flag acceptance must be
#    close to 1. This is the control that the mask is really applied to the draft and not just changing its timing.
# Tests 1-2 pin --no-spec-draft-validate. That flag (default ON since 9b77a687d, row S69-VLLM-W2-GRAMMAR-VALIDATED-DRAFTS-
# 2026-10-27-01) cuts every draft to its grammar-valid prefix BEFORE draft_n is counted, so with it the "off" run only ever
# reports grammar-legal drafts and its acceptance reads 1.0 (MEASURED gate b1019p 2026-10-27: off 17/17, on 33/104). Tests
# 1-2 measure what the DRAFT MODEL proposes, which is the thing --spec-draft-grammar changes, so they need the untrimmed count.
#  - test 3, the default path (validator on): acceptance RATE is no longer the signal, accepted TOKENS are -- the masked
#    draft keeps proposing past the point where the validator cuts an unmasked one. MEASURED b1019p: 17 -> 33 accepted.
import glob
import os

import pytest
from utils import ServerPreset, download_file

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"
GRAMMAR = 'root ::= [a-e ]+'


def target_file():
    """The target's own GGUF in the HF cache the server downloads -hf models into (LLAMA_CACHE)."""
    cache = os.environ.get("LLAMA_CACHE", "")
    hits = glob.glob(os.path.join(cache, "models--ggml-org--stories15M_MOE", "snapshots", "*", "stories15M_MOE-F16.gguf"))
    return hits[0] if hits else None


def make_server(draft_grammar, model_draft, draft_validate=False):
    server = ServerPreset.stories15m_moe()
    server.model_draft = model_draft
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 1
    server.spec_draft_n_max = 8
    server.fa = "off"
    server.temperature = 0.0
    server.extra_args = ["--spec-draft-validate" if draft_validate else "--no-spec-draft-validate"]
    if draft_grammar:
        server.extra_args.append("--spec-draft-grammar")
    return server


def run(draft_grammar, model_draft, draft_validate=False):
    server = make_server(draft_grammar, model_draft, draft_validate)
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "n_predict": 48,
        "grammar": GRAMMAR,
        "return_tokens": True,
    })
    assert res.status_code == 200
    server.stop()
    t = res.body["timings"]
    print("draft_grammar=%s draft=%s: draft_n=%d draft_n_accepted=%d content=%r" % (
        draft_grammar, os.path.basename(model_draft), t["draft_n"], t["draft_n_accepted"], res.body["content"]))
    return res.body["tokens"], t["draft_n"], t["draft_n_accepted"]


def rates(model_draft):
    tokens_off, n_off, acc_off = run(False, model_draft)
    tokens_on, n_on, acc_on = run(True, model_draft)
    assert tokens_on == tokens_off          # lossless: the target decides every token in both runs
    assert n_off > 0 and n_on > 0
    rate_off, rate_on = acc_off / n_off, acc_on / n_on
    print("acceptance off %.3f on %.3f" % (rate_off, rate_on))
    return rate_off, rate_on


def test_draft_grammar_raises_acceptance_lossless():
    rate_off, rate_on = rates(download_file(MODEL_DRAFT_FILE_URL))
    # greedy on CPU is deterministic; the measured 0.076 -> 0.354 is 4.7x. A 3x bound leaves margin and is still far
    # above what a timing-only change could give.
    assert rate_on > rate_off + 0.2
    assert rate_on >= 3 * rate_off


def test_draft_grammar_same_model_draft_accepts_nearly_all():
    path = target_file()
    if path is None:
        pytest.skip("stories15M_MOE-F16.gguf not in LLAMA_CACHE")
    rate_off, rate_on = rates(path)
    assert rate_on >= 0.9
    assert rate_on > rate_off + 0.2


def test_draft_grammar_still_adds_accepted_tokens_with_validator_on():
    draft = download_file(MODEL_DRAFT_FILE_URL)
    tokens_off, n_off, acc_off = run(False, draft, draft_validate=True)
    tokens_on, n_on, acc_on = run(True, draft, draft_validate=True)
    assert tokens_on == tokens_off          # lossless with both features on
    print("validator on: accepted off %d on %d" % (acc_off, acc_on))
    # measured 17 -> 33; 1.5x leaves margin and is far above run-to-run noise (greedy, deterministic)
    assert acc_on >= 1.5 * acc_off
