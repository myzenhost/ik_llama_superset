# Served check of fastllm C5 and C8 (commit 215461af1): the chat parser changes reach the HTTP response.
# Rows S69-FASTLLM-C5-C8-C9-SERVED-E2E-AT-GATE-2026-11-07-01, S69-FASTLLM-TOOLCALL-TRUNCATION-RECOVERY-C5-2026-10-27-01,
# S69-FASTLLM-THINKING-BOUNDARY-MENTION-AND-EMPTY-REPLY-C8-2026-10-27-01.
#
# SOURCE-READ: common/chat.cpp chat_mask_mentioned_thinking_end (a tag touching a backtick, or inside an open fence, or followed by more
# text on its line is a mention), common_chat_is_empty_thinking_reply, chat_drop_incomplete_tool_call; tools/server/server-schema.cpp
# block_level_thinking_end / error_on_empty_thinking_reply request fields; tools/server/server-task.cpp invalid_thinking_block;
# tests/test-chat.cpp test_cut_calls_other_formats / test_thinking_end_mention / test_empty_thinking_reply (the unit-level twins).
#
# DETERMINISM-SETUP: the tiny stories260K model cannot write chat text by itself, so the text is forced: C8 uses a request grammar that
# spells the exact reply (no tools, so a custom grammar is allowed). C5 is out of reach for this model (see the note at the end).
# temperature 0, seed 1, cache_prompt false.
# NULL-RUN-CONTROL: C8 runs each text twice, block_level_thinking_end true (default) and false (the pre-change parse); the false arm
# must show the old defect or the setup is not exercising the feature.
import json
import os

import pytest
from utils import ServerPreset, ServerProcess, use_local_stories260k  # noqa: E402

PORT_BASE = int(os.environ.get("LLAMA_SERVER_TEST_PORT_BASE") or os.environ.get("PORT") or "8080")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": ["Paris"]}}, "required": ["city"]},
    },
}]
MESSAGES = [{"role": "user", "content": "weather?"}]


def _gbnf_literal(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _server(template: str) -> ServerProcess:
    srv = ServerPreset.tinyllama2()
    use_local_stories260k(srv)
    srv.n_slots = 1
    srv.n_ctx = 2048
    srv.jinja = True
    srv.chat_template_file = f"../../../models/templates/{template}"
    srv.temperature = 0.0
    srv.seed = 1
    return srv


# ---- C8 ------------------------------------------------------------------------------------------------------------------------

MENTION_CASES = [
    # (what, forced text after the open <think>, reasoning, content)
    ("tag in backticks", "Quote the tag `</think>` here.\n</think>\n\nThe answer.", "Quote the tag `</think>` here.", "The answer."),
    ("tag bare in a sentence", "the closing </think> tag is special.\n</think>\n\nDone.", "the closing </think> tag is special.", "Done."),
    ("tag inside a code fence", "Example:\n```\n</think>\n```\nok\n</think>\n\nFinal.", "Example:\n```\n</think>\n```\nok", "Final."),
    ("control: no mention", "plain\n</think>\n\nAnswer", "plain", "Answer"),
]


def _chat(srv: ServerProcess, forced: str, **extra):
    body = {
        "messages": MESSAGES,
        "grammar": f"root ::= {_gbnf_literal(forced)}",
        "max_tokens": 200,
        "temperature": 0,
        "seed": 1,
        "cache_prompt": False,
    }
    body.update(extra)
    return srv.make_request("POST", "/v1/chat/completions", data=body, timeout=60)


@pytest.fixture()
def c8_server():
    srv = _server("Qwen3.5-4B.jinja")
    srv.server_port = PORT_BASE
    srv.start()
    yield srv
    srv.stop()


@pytest.mark.parametrize("what,forced,reasoning,content", MENTION_CASES)
def test_end_tag_mention_does_not_end_thinking(c8_server, what, forced, reasoning, content):
    res = _chat(c8_server, forced, block_level_thinking_end=True)
    assert res.status_code == 200, res.body
    msg = res.body["choices"][0]["message"]
    assert (msg.get("reasoning_content") or "").strip() == reasoning, (what, msg)
    assert (msg.get("content") or "").strip() == content, (what, msg)


@pytest.mark.parametrize("what,forced,reasoning,content", MENTION_CASES)
def test_end_tag_mention_null_run_flag_off(c8_server, what, forced, reasoning, content):
    """Fail-before: block_level_thinking_end=false is the pre-change parse; the mention cases must differ from the fixed result."""
    res = _chat(c8_server, forced, block_level_thinking_end=False)
    assert res.status_code == 200, res.body
    msg = res.body["choices"][0]["message"]
    got = ((msg.get("reasoning_content") or "").strip(), (msg.get("content") or "").strip())
    if what.startswith("control"):
        assert got == (reasoning, content), (what, msg)  # no mention: both parses agree
    else:
        assert got != (reasoning, content), f"null run: flag off must show the old early end of thinking, got {msg}"


def test_empty_reply_after_thinking_is_a_retryable_error(c8_server):
    forced = "r\n</think>\n\n"
    res = _chat(c8_server, forced)
    assert res.status_code != 200, f"the model ended itself with reasoning and no reply: want an error, got {res.body}"
    assert "invalid_thinking_block" in json.dumps(res.body), res.body
    # null run: the switch off returns the empty reply as before
    off = _chat(c8_server, forced, error_on_empty_thinking_reply=False)
    assert off.status_code == 200, off.body
    msg = off.body["choices"][0]["message"]
    assert (msg.get("content") or "").strip() == "", msg


def test_empty_reply_control_a_cut_turn_is_not_an_error(c8_server):
    # same reasoning, cut by max_tokens before the model ends itself: finish_reason length, no error
    res = _chat(c8_server, "a long reasoning that never closes the block " * 8, max_tokens=8)
    assert res.status_code == 200, res.body
    assert res.body["choices"][0]["finish_reason"] == "length", res.body["choices"][0]


# ---- C5 ------------------------------------------------------------------------------------------------------------------------


# C5 (tool-call truncation) is NOT covered here: with stories260K the served Qwen3-Coder grammar is
# root ::= generation-prompt space until-82 space tool-call-root, where until-82 accepts free prose before "<tool_call>", so the
# tiny model writes prose up to max_tokens and never starts a call (probe tmp/qwork_qworker-fork27/c5p_full.txt,
# 0 tool_calls under tool_choice required and named function). The unit-level twin is tests/test-chat.cpp test_cut_calls_other_formats.
