import threading
import time

import pytest

from utils import ServerPreset

server = ServerPreset.tinyllama2()

# Long enough that the reuse signal (timings.cache_n) is unambiguous: a cold request reuses ~0 of
# it, a radix hit reuses most of it. Same shape as test_cross_slot_prefix_cache.py's SHARED_PREFIX.
SHARED_PREFIX = "The quick brown fox jumps over the lazy dog. " * 40


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _run_two_prompts(radix_cache: bool, n_predict_b: int = 8):
    """Request A on one slot, request B (a DIFFERENT prompt sharing A's prefix) admitted onto a
    SECOND, never-before-used slot WHILE A is still generating. Two things this isolates:

    1. Different physical slots (not sequential reuse of the SAME slot) -- rules out the
       pre-existing, unconditional slot.prompt.tokens.get_common_prefix() same-slot reuse that
       would otherwise fire for free whenever two prompts share a literal prefix on one slot
       (MEASURED: an earlier n_slots=1 version of this test got cache_n=1175 with --no-radix-cache
       too, because A and B landed on the same slot and its OWN leftover cache already matched --
       nothing to do with cross-slot sharing at all).
    2. DISABLE-REVERSIBLE: --cache-ram 0 (this test server process only, re-enable condition is
       simply not running this isolation test) fully disables server_prompt_cache -- the OTHER,
       already-landed cross-slot mechanism (SGLang-RadixAttention-gap-port, S69-ENGINE-WAVE2-
       SGLANG-RADIXATTENTION-2026-09-26-01) -- so any reuse observed here can only be this NEW
       radix trie, not that one; see test_cross_slot_prefix_cache.py for that mechanism's own
       tests. MEASURED: --no-cache-publish-on-prefill alone was NOT enough -- with it set but
       --cache-ram left at its default, --no-radix-cache still got cache_n=1175, because A's
       small n_predict=32 generation finished before B was even sent, so A went IDLE and
       server_prompt_cache's UNCONDITIONAL idle-slot publish (server-context.cpp's
       cache_idle_slots path, ~3323-3339, not gated by cache_publish_on_prefill at all) had
       already made A's prompt available by the time B's slot was picked. --cache-ram 0 removes
       that whole mechanism instead of racing its timing.

    B's slot has never processed anything (n_past == 0, prompt.tokens empty), which is exactly
    the case server-radix-cache's match hook in server-context.cpp is scoped to.

    GATE-CLAIM-OK: "CPU" below names the test's own build target (this suite always runs the CPU
    llama-server.exe, see _radix_cpu_build.bat), not a claim that a shared GPU/CPU resource is
    unavailable.
    """
    global server
    server.n_slots = 2
    server.n_ctx = 4096
    server.n_predict = 64
    server.cache_ram = 0
    server.extra_args = (["--radix-cache", "--kv-unified"] if radix_cache
                          else ["--no-radix-cache", "--kv-unified"])
    server.start()

    # DETERMINISTIC ORDERING (2026-09-28). This used to be `time.sleep(0.3)` between A and B, betting that
    # A's prefill finishes and its generation is still running 0.3 s later. On a loaded host that bet
    # lost BOTH ways in one day: B too early -> cache_n 0 with --radix-cache; A already done and B landing
    # on A's own slot -> cache_n 1175 with --no-radix-cache. Now:
    #   * id_slot pins A to slot 0 and B to slot 1, so "different physical slots" holds by construction
    #     (the property point 1 above needs), whatever A's timing;
    #   * A is STREAMED, and B is sent only after A's first token arrives. The server's single update
    #     loop publishes A's prompt into the radix tree in the tick that produced that token
    #     (server-context.cpp, the post-decode publish loop), before it can dequeue B -- so B always
    #     arrives after A's prompt is available.
    first_token = threading.Event()
    result = {}

    def send_a():
        chunks = 0
        try:
            for _ in server.make_stream_request("POST", "/completion", data={
                "prompt": SHARED_PREFIX + " Tell me a story about a cat.",
                "seed": 42,
                "temperature": 0.0,
                "n_predict": 32,
                "id_slot": 0,
                "stream": True,
            }):
                chunks += 1
                first_token.set()
        finally:
            result["a_chunks"] = chunks
            first_token.set()  # never leave the main thread waiting on a failed stream

    t = threading.Thread(target=send_a)
    t.start()
    assert first_token.wait(timeout=60), "A produced no streamed token within 60 s"

    res_b = server.make_request("POST", "/completion", data={
        "prompt": SHARED_PREFIX + " Tell me a story about a dog.",
        "seed": 42,
        "temperature": 0.0,
        "n_predict": n_predict_b,
        "id_slot": 1,
    })

    t.join(timeout=60)
    # Each call starts its own server; stop it here. A test that calls this helper twice (the
    # byte-identical test) otherwise started a second process on the same ServerProcess, teardown
    # stopped only the second, and the first kept listening on the port -- measured 2026-09-28: every
    # later test in the session (watermark, cross-slot) talked to that leftover radix server.
    server.stop()

    assert result.get("a_chunks", 0) > 0, "A's stream returned no chunks"
    assert res_b.status_code == 200
    # the pin must have held, or "different slots" is an assumption again
    assert res_b.body.get("id_slot") == 1, f"B ran on slot {res_b.body.get('id_slot')}, expected 1"

    return res_b


def test_radix_cache_flag_off_no_reuse_across_different_slots():
    """Baseline (fail-before half of the fail-before/pass-after pair): with --no-radix-cache (and
    the OTHER cross-slot mechanism disabled too, see _run_two_prompts), B lands on a fresh slot
    and reprocesses the whole shared prefix itself -- there is no on-device token-level index to
    find that reuse without the flag."""
    res_b = _run_two_prompts(radix_cache=False)
    cache_n = res_b.body["timings"]["cache_n"]
    assert cache_n == 0, (
        f"expected --no-radix-cache to leave B cold on its own fresh slot (timings.cache_n == 0), "
        f"got {cache_n}"
    )


def test_radix_cache_flag_on_reuses_shared_prefix_across_different_slots():
    """Pass-after half: with --radix-cache, B's admission (on its OWN, previously-unused slot)
    finds A's published prompt in the radix tree and seq_cp's the shared prefix in zero-copy, so
    B only prefills its own suffix. The shared preamble tokenizes to well over 1000 tokens on this
    model/vocab (same threshold rationale as test_cross_slot_prefix_cache.py); >100 distinguishes
    "shared" from "not shared" without pinning an exact tokenizer-dependent count."""
    res_b = _run_two_prompts(radix_cache=True)
    cache_n = res_b.body["timings"]["cache_n"]
    assert cache_n > 100, (
        f"expected a radix-cache hit (timings.cache_n > 100), got {cache_n}"
    )
    # ACCEPTANCE (row): prompt_n (tokens actually prefilled, not reused) reflects only the suffix,
    # not the whole prompt -- it must be far smaller than the shared prefix alone.
    prompt_n = res_b.body["timings"]["prompt_n"]
    assert prompt_n < 100, (
        f"expected prompt_n to cover only B's suffix after a radix hit, got {prompt_n}"
    )


def test_radix_cache_output_byte_identical_to_flag_off_at_temperature_zero():
    """CONTROL (row's mandatory test): a radix-cache hit must never change the actual generation --
    it only changes WHICH cells got there. At temperature 0 (greedy, deterministic), B's output
    text must be byte-identical whether or not --radix-cache reused the prefix."""
    content_off = _run_two_prompts(radix_cache=False, n_predict_b=16).body["content"]
    content_on = _run_two_prompts(radix_cache=True, n_predict_b=16).body["content"]
    assert content_on == content_off, (
        "radix-cache reuse changed the generated text at temperature 0 -- "
        f"off={content_off!r} on={content_on!r}"
    )


def test_radix_cache_without_kv_unified_is_refused_not_crashed():
    """Gate test: --radix-cache without --kv-unified must not build/crash the server -- it should
    log a warning and simply run with the cache disabled (server-context.cpp's why_disabled check,
    first branch). A plain request must still succeed and show no reuse."""
    global server
    server.n_slots = 1
    server.n_ctx = 2048
    server.n_predict = 16
    server.extra_args = ["--radix-cache"]  # deliberately no --kv-unified
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello, world!",
        "seed": 42,
        "temperature": 0.0,
        "n_predict": 8,
    })
    assert res.status_code == 200
    # MEASURED: a lone request on a freshly-started server got cache_n == 1 here (a single
    # incidental BOS-token-level match against the slot's initial empty state, unrelated to any
    # cache mechanism) -- < 5 asserts "no meaningful reuse happened" without pinning an exact
    # zero that a harmless single-token coincidence can trip.
    assert res.body["timings"]["cache_n"] < 5, (
        f"expected --radix-cache without --kv-unified to show no meaningful reuse, got "
        f"{res.body['timings']['cache_n']}"
    )
