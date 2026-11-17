# Served check of fastllm C9 (commit 63e817644): --prefill-ubatch-chunks, a cancelled prefill is dropped after one micro-batch.
# Row S69-FASTLLM-C5-C8-C9-SERVED-E2E-AT-GATE-2026-11-07-01, S69-FASTLLM-CANCEL-LONG-PREFILL-C9-2026-10-27-01.
#
# SOURCE-READ: tools/server/server-context.cpp:8835-8839 (prefill_cap_this is capped at n_ubatch when the flag is on and the slot can
# split its prompt), common/arg.cpp:2431-2437 (the flag, env LLAMA_ARG_PREFILL_UBATCH_CHUNKS), server-context.cpp:1267-1268 (/slots
# reports n_prompt_tokens and n_prompt_tokens_processed).
#
# Scenario: one slot, n_batch 2048 / n_ubatch 8, a long prompt. The client closes the socket while the prompt is being prefilled, then a
# short request is sent. NO WALL-CLOCK THRESHOLDS (they flake under host load): the test asserts behaviour only.
#   1. at the moment of the disconnect the prefill is partial: 0 < n_prompt_tokens_processed < n_prompt_tokens (control: the cancel really
#      lands mid-prefill; a slower host only makes this more true),
#   2. the slot reports idle after the cancel (the only wait is a generous 60 s safety limit, not a performance bound),
#   3. /slots shows the prefill stopped before the full prompt: n_prompt_tokens_processed < n_prompt_tokens after the slot went idle,
#   4. the next request returns 200 with the same content as on an idle server.
# The flag-off arm is the null run: the fork always cancelled, only one scheduler step later, so it must satisfy the same behaviour.
# DETERMINISM-SETUP: token-id prompts, temperature 0, seed 1, cache_prompt false, one pinned slot; the short answer is compared to a
# solo run on the same server.
import http.client
import json
import time

import pytest

from utils import ServerPreset, use_local_stories260k  # noqa: E402

LONG_TOKENS = 1900  # the server clamps the usable context to n_ctx_train (2048) of the tiny model
SHORT_TOKENS = 20
IDLE_SAFETY_LIMIT_S = 60.0  # a hang detector far above any measured value (a few seconds), not a speed bound


def _make_server(chunks: bool):
    srv = ServerPreset.tinyllama2()
    use_local_stories260k(srv)
    srv.n_slots = 1
    srv.server_slots = True
    srv.n_ctx = 2048
    srv.n_batch = 2048
    srv.n_ubatch = 8
    srv.extra_args = ["--prefill-ubatch-chunks"] if chunks else []
    return srv


def _prompt(n, base):
    return [1] + [3 + ((i * 7 + base) % 500) for i in range(n - 1)]


def _body(prompt):
    return {"prompt": prompt, "n_predict": 4, "temperature": 0.0, "seed": 1, "cache_prompt": False}


def _slot(srv):
    return srv.make_request("GET", "/slots").body[0]


def test_cancel_mid_prefill_frees_the_slot_and_the_next_request_is_served_flag_on():
    _run(True)


def test_cancel_mid_prefill_frees_the_slot_and_the_next_request_is_served_flag_off():
    _run(False)


def _run(chunks):
    srv = _make_server(chunks)
    srv.start()
    solo = srv.make_request("POST", "/completion", data=_body(_prompt(SHORT_TOKENS, 11)))
    assert solo.status_code == 200 and solo.body["content"], solo.body

    conn = http.client.HTTPConnection(srv.server_host, srv.server_port, timeout=30)
    conn.request("POST", "/completion", body=json.dumps(_body(_prompt(LONG_TOKENS, 0))), headers={"Content-Type": "application/json"})
    seen = None
    if chunks:
        # flag on: one micro-batch per scheduler step, so /slots answers between steps; wait until some prompt tokens are
        # processed, then leave at once. Flag off: the whole 1900-token prompt is ONE scheduler step (n_batch 2048), /slots cannot
        # answer inside it and the cancel can only land after it, so this observation is the flag's own effect (RED without the flag).
        deadline = time.time() + IDLE_SAFETY_LIMIT_S
        while time.time() < deadline:
            s = _slot(srv)
            if s.get("is_processing") and s.get("n_prompt_tokens_processed", 0) > 0:
                seen = s
                break
            time.sleep(0.002)
        assert seen is not None, "the long request never showed a partial prefill (flag on must split the prompt into micro-batches)"
        assert seen["n_prompt_tokens_processed"] < seen["n_prompt_tokens"], f"control: prefill already complete when the client left: {seen}"
    else:
        time.sleep(0.3)
    conn.close()

    deadline = time.time() + IDLE_SAFETY_LIMIT_S
    after = _slot(srv)
    while after.get("is_processing") and time.time() < deadline:
        time.sleep(0.005)
        after = _slot(srv)
    assert not after.get("is_processing"), f"the cancelled prefill never released the slot: {after}"
    if chunks:
        assert after["n_prompt_tokens_processed"] < after["n_prompt_tokens"], f"prefill ran to the end of the prompt despite the cancel: {after}"

    nxt = srv.make_request("POST", "/completion", data=_body(_prompt(SHORT_TOKENS, 11)), timeout=60)
    assert nxt.status_code == 200 and nxt.body["content"], nxt.body
    assert nxt.body["content"] == solo.body["content"], "the next request after a cancel must give the idle-server answer"
    print(f"[c9] chunks={chunks} at cancel {seen and seen['n_prompt_tokens_processed']}, after idle {after.get('n_prompt_tokens_processed')}/{after.get('n_prompt_tokens')}")
