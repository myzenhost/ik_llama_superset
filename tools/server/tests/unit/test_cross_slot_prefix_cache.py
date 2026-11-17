import threading
import time

import pytest

from utils import ServerPreset

server = ServerPreset.tinyllama2()

# A long-ish shared preamble so the cache-hit signal (tokens_cached) is unambiguous: without
# cross-slot sharing this is 0 (or near it) on a cold slot, with it this is most of the preamble.
SHARED_PREFIX = "The quick brown fox jumps over the lazy dog. " * 40


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


# A's generation must still be running when the sharers arrive: an IDLE A gets published through the
# unconditional idle-slot cache path, which would make the gate-off test see reuse for the wrong
# reason. 512 tokens with ignore_eos keeps A busy far longer than one HTTP round trip.
A_N_PREDICT = 512


def _start_a_streaming(prompt, id_slot=0):
    """Start request A STREAMED on a pinned slot; return (thread, result, first_token_event).

    DETERMINISTIC ORDERING (2026-09-28), replacing `time.sleep(0.3)`: the sharers are sent only after
    A's first streamed token arrives. The server's single update loop publishes A's prompt in the tick
    that produced that token, before it can dequeue the next request, so a sharer can never arrive
    before the publish -- which the sleep only guessed at, and lost on a loaded host."""
    first = threading.Event()
    result = {}

    def run():
        chunks = 0
        try:
            for _ in server.make_stream_request("POST", "/completion", data={
                "prompt": prompt,
                "seed": 42,
                "temperature": 0.0,
                "n_predict": A_N_PREDICT,
                "ignore_eos": True,
                "id_slot": id_slot,
                "stream": True,
            }):
                chunks += 1
                first.set()
        finally:
            result["a_chunks"] = chunks
            first.set()

    t = threading.Thread(target=run)
    t.start()
    assert first.wait(timeout=60), "A produced no streamed token within 60 s"
    return t, result


def test_concurrent_request_reuses_other_slots_shared_prefix():
    """SGLang-RadixAttention gap port (S69-ENGINE-WAVE2-SGLANG-RADIXATTENTION-2026-09-26-01):
    two DIFFERENT concurrent requests sharing a long prefix, landing on two DIFFERENT slots.
    Request A starts first and is still busy (prompt done, generating) when request B is admitted.
    Without cross-slot sharing, B's slot is cold: it must reprocess the whole shared prefix itself
    (tokens_cached ~= 0). With it, B's admission finds A's already-published prefix in the shared
    prompt cache and restores it, so most of the prefix comes back as a cache hit.
    """
    global server
    server.n_slots = 2
    server.n_ctx = 4096
    server.n_predict = 64
    server.start()

    t, result = _start_a_streaming(SHARED_PREFIX + " Tell me a story about a cat.")

    # best-effort evidence that A is genuinely still busy (concurrency, not eventual-idle reuse) --
    # informational only, not asserted (and never allowed to fail the test): exact timing on a
    # shared CI/dev host is not guaranteed, and the /slots endpoint itself is off unless the
    # preset enables it.
    n_busy_mid = None
    try:
        slots_mid = server.make_request("GET", "/slots")
        if isinstance(slots_mid.body, list):
            n_busy_mid = sum(1 for s in slots_mid.body if s["is_processing"])
    except Exception:
        pass

    res_b = server.make_request("POST", "/completion", data={
        "prompt": SHARED_PREFIX + " Tell me a story about a dog.",
        "seed": 42,
        "temperature": 0.0,
        "n_predict": 8,
        "id_slot": 1,
    })

    t.join(timeout=120)

    assert result.get("a_chunks", 0) > 0
    assert res_b.status_code == 200
    assert res_b.body.get("id_slot") == 1, f"B ran on slot {res_b.body.get('id_slot')}, expected 1"

    # NOTE: the top-level "tokens_cached" field (server_task_result_cmpl_final::n_tokens_cached,
    # tools/server/server-context.cpp:2393) is slot.prompt.n_tokens() -- the TOTAL size of the
    # slot's cache AFTER this request, which is nonzero for every request regardless of reuse and
    # is NOT a reuse signal. The actual reuse count is timings.cache_n
    # (server_slot_stats::to_json(), tools/server/server-common.cpp:69, backed by
    # slot.stats.n_prompt_cached = n_past at tools/server/server-context.cpp:3668) -- MEASURED:
    # using "tokens_cached" here made this test pass even against the unpatched base commit
    # (it was just reporting slot 1's post-request total size, ~1178 prompt + a few generated
    # tokens, not a cache hit), while "timings.cache_n" correctly read back 0 on that same run.
    #
    # The shared preamble tokenizes to well over 1000 tokens on this model/vocab; a genuinely cold
    # slot (no cross-slot sharing) reuses 0 of it. A conservative threshold well below the full
    # preamble length, but far above what any incidental single-token coincidence could produce,
    # distinguishes "shared" from "not shared" without pinning an exact tokenizer-dependent count.
    cache_n = res_b.body["timings"]["cache_n"]
    assert cache_n > 100, (
        f"expected cross-slot prefix reuse (timings.cache_n > 100), got "
        f"{cache_n} (mid-run busy slots: {n_busy_mid})"
    )


def test_three_way_concurrent_prefix_sharing_non_consuming_load():
    """N-consumer follow-up (S69-ENGINE-WAVE2-SGLANG-RADIXATTENTION-NCONSUMER-FOLLOWUP-2026-09-27-01).
    Request A publishes a shared prefix on prefill and keeps generating. Then B and C -- two MORE
    requests sharing that prefix -- are admitted at the same moment, on two more slots. Before the
    fix, server_prompt_cache::load() erased the entry it served, so whichever of B/C was admitted
    first consumed it and the other found nothing (its own prefix was not published yet, since
    neither had prefilled). With the non-consuming load both must hit.

    B and C are released together from a barrier so they are admitted in the same scheduler pass;
    several rounds with DISTINCT prefixes make a timing coincidence (C arriving after B's own
    publish) unable to pass every round on the old behaviour.
    """
    global server
    server.n_slots = 3
    server.n_ctx = 8192
    server.n_predict = 64
    server.start()

    misses = []
    for rnd in range(4):
        prefix = f"Round {rnd}: " + ("The quick brown fox jumps over the lazy dog. " * 40)
        result = {}

        def send(key, suffix, n_predict, id_slot):
            result[key] = server.make_request("POST", "/completion", data={
                "prompt": prefix + suffix,
                "seed": 42,
                "temperature": 0.0,
                "n_predict": n_predict,
                "id_slot": id_slot,
            })

        # A streamed on slot 0; B and C released only after A's first token (prefill done + published)
        ta, a_result = _start_a_streaming(prefix + " Tell me a story about a cat.", id_slot=0)

        barrier = threading.Barrier(2)

        def gated(key, suffix, id_slot):
            barrier.wait()
            send(key, suffix, 8, id_slot)

        tb = threading.Thread(target=gated, args=("b", " Tell me a story about a dog.", 1))
        tc = threading.Thread(target=gated, args=("c", " Tell me a story about a bird.", 2))
        tb.start(); tc.start()
        tb.join(timeout=60); tc.join(timeout=60); ta.join(timeout=120)

        assert a_result.get("a_chunks", 0) > 0
        for key, want_slot in (("b", 1), ("c", 2)):
            assert result[key].status_code == 200
            assert result[key].body.get("id_slot") == want_slot, (
                f"round {rnd}: {key} ran on slot {result[key].body.get('id_slot')}, expected {want_slot}")
        for key in ("b", "c"):
            cache_n = result[key].body["timings"]["cache_n"]
            if cache_n <= 100:
                misses.append((rnd, key, cache_n))

    assert not misses, (
        f"expected every concurrently-admitted sharer to reuse the published prefix "
        f"(timings.cache_n > 100); misses (round, request, cache_n): {misses}"
    )


def test_cache_publish_on_prefill_flag_off_disables_cross_slot_reuse():
    """Gate test for --no-cache-publish-on-prefill (S69-ENGINE-WAVE2-SGLANG-RADIXATTENTION-
    2026-09-26-01 parent-review cost finding). Identical shape to
    test_concurrent_request_reuses_other_slots_shared_prefix above, except the server is started
    with the gate OFF: the proactive publish at SLOT_STATE_DONE_PROMPT must not fire, so B's slot
    stays cold and reprocesses the whole shared prefix itself (timings.cache_n == 0). This is the
    fail-before/pass-after case for the GATE itself, not for the underlying feature: the feature's
    own before/after is test_concurrent_request_reuses_other_slots_shared_prefix (0 on unpatched
    base, >100 on this branch with the flag at its default-on) -- this test would itself go from
    PASS on a build that lacks the flag entirely (nothing to disable) to a meaningful PASS/FAIL only
    once --no-cache-publish-on-prefill exists and is wired to the publish loop, which is what it
    verifies.
    """
    global server
    server.n_slots = 2
    server.n_ctx = 4096
    server.n_predict = 64
    server.no_cache_publish_on_prefill = True
    server.start()

    t, result = _start_a_streaming(SHARED_PREFIX + " Tell me a story about a cat.")

    res_b = server.make_request("POST", "/completion", data={
        "prompt": SHARED_PREFIX + " Tell me a story about a dog.",
        "seed": 42,
        "temperature": 0.0,
        "n_predict": 8,
        "id_slot": 1,
    })

    t.join(timeout=120)

    assert result.get("a_chunks", 0) > 0
    assert res_b.status_code == 200
    assert res_b.body.get("id_slot") == 1, f"B ran on slot {res_b.body.get('id_slot')}, expected 1"

    cache_n = res_b.body["timings"]["cache_n"]
    assert cache_n == 0, (
        f"expected the gate OFF to suppress cross-slot publish (timings.cache_n == 0), got "
        f"{cache_n} -- the flag is not disabling the proactive publish"
    )
