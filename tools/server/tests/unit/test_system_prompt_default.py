"""Server-wide default system prompt: --system-prompt / --system-prompt-file / the fork's -spf.

The fork's pre-merge server accepted -spf and prefixed that prompt to every slot; the upstream merge 279b4bfaf
dropped the flag and upstream enabled --system-prompt(-file) for the CLI tools only. Restored for llama-server as
a system MESSAGE inserted into chat requests that bring none of their own (the request always wins).
Checked through /apply-template, which renders the exact prompt the chat endpoints would send.
"""
import os
import tempfile

import pytest

from utils import ServerPreset

server = ServerPreset.tinyllama2()

SYS = "You are the fork default."
CR = "<|START_OF_TURN_TOKEN|>"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.chat_template = "command-r"


def _render(messages):
    res = server.make_request("POST", "/apply-template", data={"messages": messages})
    assert res.status_code == 200, res.body
    return res.body["prompt"]


def test_no_flag_inserts_nothing():
    # control: without the flag the prompt has no system turn at all
    server.start()
    prompt = _render([{"role": "user", "content": "Hi there"}])
    assert "<|SYSTEM_TOKEN|>" not in prompt
    assert SYS not in prompt


@pytest.mark.parametrize("flag", ["--system-prompt", "-sys"])
def test_system_prompt_flag_inserted_when_request_has_none(flag):
    server.extra_args = [flag, SYS]
    server.start()
    prompt = _render([{"role": "user", "content": "Hi there"}])
    assert prompt == (f"{CR}<|SYSTEM_TOKEN|>{SYS}<|END_OF_TURN_TOKEN|>"
                      f"{CR}<|USER_TOKEN|>Hi there<|END_OF_TURN_TOKEN|>{CR}<|CHATBOT_TOKEN|>")


@pytest.mark.parametrize("flag", ["--system-prompt-file", "-sysf", "-spf"])
def test_system_prompt_file_flags(flag):
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(SYS + "\n")  # trailing newline is stripped by the parser
    server.extra_args = [flag, path]
    server.start()
    prompt = _render([{"role": "user", "content": "Hi there"}])
    assert f"<|SYSTEM_TOKEN|>{SYS}<|END_OF_TURN_TOKEN|>" in prompt


def test_request_system_message_wins():
    server.extra_args = ["--system-prompt", SYS]
    server.start()
    prompt = _render([{"role": "system", "content": "Request's own."}, {"role": "user", "content": "Hi there"}])
    assert "Request's own." in prompt
    assert SYS not in prompt
    assert prompt.count("<|SYSTEM_TOKEN|>") == 1
