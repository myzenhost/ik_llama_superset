# --prefill-fair-share (row S69-VLLM-PICK-FAIR-PREFILL-CHUNK-SHARE-2026-09-29-01).
#
# SOURCE-READ: engines/vllm/vllm/v1/core/sched/scheduler.py:606-621 (a long prefill is capped only while other requests are eligible;
# the cap floors at budget / eligible) and engines/exllamav2/generator/dynamic.py:982-985 (each active job prefills its own chunk every
# round). Apache-2.0 / MIT; behaviour reimplemented, no code copied.
#
# Setup: two slots, a tiny batch, one very long prompt already prefilling when a 20-token prompt arrives. Without the flag the first
# slot takes the whole batch at every step, so the short request is answered after the long prefill finishes. With it the short prompt
# takes what it needs in the next step and is answered first. Output bytes of the short request must equal a solo run (temperature 0).
# DETERMINISM-SETUP: token-id prompts, temperature 0, seed 1, cache_prompt false, pinned slots; the flag-off arm also compares its short answer to a solo run.
# NULL-RUN-CONTROL: test_without_the_flag_the_short_prompt_waits_for_the_long_one (feature off, same concurrent pair, same solo comparison) must PASS.
import threading
import time

import pytest

from utils import ServerPreset, use_local_stories260k  # noqa: E402

server = ServerPreset.tinyllama2()

LONG_TOKENS = 1900
SHORT_TOKENS = 20


def _make_server(fair: bool):
    srv = ServerPreset.tinyllama2()
    use_local_stories260k(srv)
    srv.n_slots = 2
    srv.server_slots = True   # /slots is how the test sees the long request start prefilling
    srv.n_ctx = 8192          # 4096 per slot
    srv.n_batch = 8
    srv.n_ubatch = 8
    srv.extra_args = ["--prefill-fair-share"] if fair else []
    return srv


def _prompt(n, base):
    # token-id prompts avoid any tokenizer variance; BOS(1) first, then ids spread over the vocabulary
    return [1] + [3 + ((i * 7 + base) % 500) for i in range(n - 1)]


def _run_pair(srv):
    """Start the long request, wait until its slot is prefilling, send the short one. Returns (t_long, t_short, short_body)."""
    done = {}
    t0 = time.time()

    def call(name, prompt, slot):
        # slots are pinned: the long prompt sits in the FIRST slot, which is the one a first-come-takes-all batch favours
        res = srv.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": 4, "temperature": 0.0, "seed": 1, "cache_prompt": False,
            "id_slot": slot}, timeout=120)
        done[name] = (time.time() - t0, res)

    th_long = threading.Thread(target=call, args=("long", _prompt(LONG_TOKENS, 0), 0))
    th_long.start()
    deadline = time.time() + 5
    started = False
    while time.time() < deadline and not started:
        slots = srv.make_request("GET", "/slots").body
        started = any(s.get("is_processing") for s in slots)
        time.sleep(0.01)
    assert started, "control: the long request must be in flight before the short one is sent"
    th_short = threading.Thread(target=call, args=("short", _prompt(SHORT_TOKENS, 11), 1))
    th_short.start()
    th_long.join(120)
    th_short.join(120)
    assert done["long"][1].status_code == 200 and done["short"][1].status_code == 200
    return done["long"][0], done["short"][0], done["short"][1].body


def _solo_short(srv):
    res = srv.make_request("POST", "/completion", data={
        "prompt": _prompt(SHORT_TOKENS, 11), "n_predict": 4, "temperature": 0.0, "seed": 1, "cache_prompt": False})
    assert res.status_code == 200
    return res.body["content"]


def test_without_the_flag_the_short_prompt_waits_for_the_long_one():
    srv = _make_server(fair=False)
    srv.start()
    t_long, t_short, body = _run_pair(srv)
    assert t_short >= t_long - 0.05, f"control: without the flag the short request should finish after the long prefill (short {t_short:.3f}s, long {t_long:.3f}s)"
    # null-run control: feature off, same concurrent pair; if this differs the setup is nondeterministic, not the flag
    assert body["content"] == _solo_short(srv), "null run: flag off, concurrent short answer must equal the solo run"


def test_with_the_flag_the_short_prompt_finishes_first_and_output_is_unchanged():
    srv = _make_server(fair=True)
    srv.start()
    t_long, t_short, body = _run_pair(srv)
    assert t_short < t_long, f"with the flag the short request finishes first (short {t_short:.3f}s, long {t_long:.3f}s)"
    assert body["content"] == _solo_short(srv), "the share only changes scheduling, never the tokens"
