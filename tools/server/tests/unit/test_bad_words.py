"""bad_words: vLLM's SamplingParams.bad_words field name, accepted as an ALIAS onto our existing
banned_strings mechanism (rewind-and-resample -- see test_banned_strings.py).

Task row S69-VLLM-SAMPLING-DELTA-5-GAPS-2026-09-14-01. Gap measured 2026-09-14 (bad_words: 0
files, full scope). Closed as an ALIAS 2026-09-26, not a byte-for-byte port: vLLM's bad_words is
TOKEN-level (SOURCE-READ: portable/envs/vllm/Lib/site-packages/vllm/v1/sample/ops/bad_words.py --
precomputes exactly 2 tokenizations of the word, with and without a leading space, and masks the
last token id of whichever one's prefix matches the tail of the tokens generated so far, BEFORE
sampling). Ours is TEXT-level (server-context.cpp: decode running text, rewind on a
case-insensitive substring match, ban that token, resample).

test_bad_words_is_alias_for_banned_strings mirrors test_banned_strings.py's own test to show
parity of effect. test_bad_words_catches_a_third_tokenization is the measured semantic
difference: it shows (via /tokenize) that the word's IN-CONTEXT token sequence, as the model
actually generated it, is neither of vLLM's two precomputed tokenizations -- a case vLLM's
mechanism (restricted to those 2 sequences) would not match, but ours (text-level) still catches.
"""
import re

import pytest
from utils import *

server = ServerPreset.tinyllama2()

PROMPT = "Once upon a time"
N_PREDICT = 48


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _complete(**extra) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": N_PREDICT, "temperature": 0, "cache_prompt": False, **extra,
    })
    assert res.status_code == 200, res.body
    return res.body


def _tokenize(content: str) -> list:
    res = server.make_request("POST", "/tokenize", data={"content": content, "add_special": False})
    assert res.status_code == 200, res.body
    return res.body["tokens"]


def _a_word(text: str) -> str:
    # the first word of 4+ letters that is not at the very start, so there is a prefix to preserve
    words = re.findall(r"[A-Za-z]{4,}", text)
    assert len(words) >= 2, text
    return words[1]


def _candidate_words(text: str) -> list:
    # every 4+-letter word after the first, in order -- used where one candidate's tokenization
    # might not satisfy a test's own precondition and a fallback is needed (BPE is not
    # prefix-stable: appending characters to a string can change how EARLIER characters split,
    # so a naive prefix/suffix token diff can come back empty for a specific word by chance).
    words = re.findall(r"[A-Za-z]{4,}", text)
    return words[1:]


def test_bad_words_is_alias_for_banned_strings():
    server.start()
    ref = _complete()["content"]
    word = _a_word(ref)
    at = ref.lower().find(word.lower())

    got = _complete(bad_words=[word])
    assert word.lower() not in got["content"].lower()
    # rewind-and-resample: text before the match is unchanged, exactly like banned_strings
    assert got["content"][:at] == ref[:at]
    # bad_words and banned_strings write the same underlying vector -- both keys echo it
    assert got["generation_settings"]["bad_words"] == [word.lower()]
    assert got["generation_settings"]["banned_strings"] == [word.lower()]


def test_bad_words_last_field_registered_wins_when_both_sent():
    # bad_words is registered AFTER banned_strings in the schema (server-schema.cpp), and both
    # clear()+overwrite the same ctx.params.ban_phrases vector, so when a request sends both,
    # whichever the eval loop reaches last (bad_words) wins. This is documented behaviour, not
    # a union -- measure it rather than assume it.
    server.start()
    ref = _complete()["content"]
    word = _a_word(ref)
    got = _complete(banned_strings=["zzz_not_present_zzz"], bad_words=[word])
    assert word.lower() not in got["content"].lower()
    assert got["generation_settings"]["bad_words"] == [word.lower()]
    assert got["generation_settings"]["banned_strings"] == [word.lower()]


def test_bad_words_catches_a_third_tokenization_vllm_would_miss():
    # vLLM's mechanism only ever precomputes 2 tokenizations of the bad word (with, and without,
    # a leading space -- sampling_params.py update_from_tokenizer) and only masks a token whose
    # PREFIX exactly matches one of those 2 sequences. Ours matches on decoded TEXT instead, so it
    # also catches the word however the model actually split it. Measure that the in-context split
    # is a genuine third case, not one of the two vLLM would have precomputed.
    #
    # BPE is not prefix-stable: appending the word's characters to the prefix can change how the
    # tokenizer splits characters BEFORE the word too, so a naive `with_word_ids[len(prefix_ids):]`
    # diff can legitimately come back empty or malformed for a specific reference word (MEASURED
    # 2026-09-26: happened for "little" against tinyllama2's tiny vocab). That is a property of
    # the word picked at random from this run's own greedy completion, not of bad_words -- so try
    # each 4+-letter candidate in turn instead of hard-failing on the first one's tokenization.
    server.start()
    ref = _complete()["content"]
    candidates = _candidate_words(ref)
    assert candidates, ref

    for word in candidates:
        at = ref.lower().find(word.lower())
        if at < 0:
            continue

        no_space_ids = _tokenize(word)
        space_ids = _tokenize(" " + word)

        # the token(s) the model actually used to spell the word mid-generation: tokenize the
        # prefix with and without the word appended, and take the tail tokens that differ.
        prefix_ids = _tokenize(ref[:at])
        with_word_ids = _tokenize(ref[:at + len(word)])
        in_context_ids = with_word_ids[len(prefix_ids):]

        if not in_context_ids:
            continue  # this word's tokenization does not give a usable in-context suffix, try the next
        if in_context_ids == no_space_ids or in_context_ids == space_ids:
            continue  # not a genuine third case for this word, try the next

        # vLLM restricted to {no_space_ids, space_ids} would not have a rule matching
        # in_context_ids. Our text-level bad_words still blocks it:
        got = _complete(bad_words=[word])
        assert word.lower() not in got["content"].lower()
        return

    pytest.fail(
        "no candidate word in this run's reference completion produced a genuine third "
        f"tokenization to test against (tried {candidates}); this is a test-setup limitation "
        "of the tiny reference model/vocab, not evidence about bad_words itself"
    )


def test_no_bad_words_is_unchanged():
    # control: an empty list must leave output byte-identical to the unbuffered path
    server.start()
    assert _complete()["content"] == _complete(bad_words=[])["content"]
