import os
import random
import pytest
from utils import ServerPreset, ServerProcess

# Context checkpoints of a spilled prompt-cache entry go to the NVMe tier too (--cache-disk-dir), row
# S69-TIERED-CKPT-BLOBS-INTO-DISK-TIER-2026-10-05-01. Unit coverage without a model: tests/probe-disk-tier.cpp,
# tests/probe-ckpt-spill.cpp, tests/test-server-prompt-cache-disk.cpp (tests 9-11).
#
# These tests need a model whose memory is checkpointed (recurrent / hybrid / SWA), because a plain transformer never creates a
# context checkpoint. They use a REAL small hybrid model: granite-4.0-h-350m (mamba + attention), found through
# TIERED_HYBRID_MODEL or the known local paths; the tests skip with the reason when none exists.
#
# Why this fails on a build without the feature (measured 2026-10-05 with the same flags on builds/tiered-1005-cpu): the old
# build spills the main state but keeps the checkpoints (43 MiB per cached prompt here) in memory, so --cache-ram fills with
# checkpoints, evict_one() logs "removing oldest spilled cache entry (checkpoints only: 43.156 MiB)" and drops the whole entry:
# the repeated prompt then re-processes all 623 tokens instead of 4.

CANDIDATES = [
    os.environ.get("TIERED_HYBRID_MODEL", ""),
    "models/gguf/granite4-nano-350m/granite-4.0-h-350m-Q8_0.gguf",
    "models/class-arch-smoke/Falcon-H1-0.5B-Instruct-Q8_0.gguf",
]

server = ServerPreset.tinyllama2()


def hybrid_model() -> str:
    for p in CANDIDATES:
        if p and os.path.isfile(p):
            return p
    pytest.skip("no small hybrid model found (set TIERED_HYBRID_MODEL to a recurrent/hybrid GGUF, e.g. granite-4.0-h-350m)")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = hybrid_model()
    server.temperature = 0.0
    server.n_slots = 1            # every new, different prompt moves the slot's old prompt into the cache
    server.n_ctx = 8192
    server.n_batch = 512          # the preset's 32 would put the checkpoints only in the last few tokens of the prompt
    server.n_ubatch = 512
    server.server_metrics = True
    server.n_predict = 2
    server.n_threads = 8


def metrics(server: ServerProcess) -> dict:
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    out = {}
    for line in res.body.splitlines():
        if line.startswith("llamacpp:") and "{" not in line:
            name, value = line.split(" ", 1)
            out[name] = float(value)
    return out


def make_prompts(n: int, n_words: int = 600) -> list:
    rng = random.Random(7)
    words = ["lily", "tom", "park", "ball", "tree", "sun", "dog", "cat", "red", "big", "happy", "ran", "jump", "play", "river",
             "stone", "cloud", "window", "garden", "yellow", "quiet", "bridge", "orange", "table", "music", "winter", "paper"]
    return [f"story {i}: " + " ".join(rng.choice(words) for _ in range(n_words)) for i in range(n)]


def complete(server: ServerProcess, prompt: str) -> dict:
    res = server.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True, "n_predict": 2})
    assert res.status_code == 200
    return res.body


def timings(body: dict) -> tuple:
    t = body["timings"]
    return t["prompt_n"], t["cache_n"]


# 120 MiB of host cache holds about one prompt of this model (a cached prompt is ~25 MB of state plus two ~22 MB checkpoints)
CACHE_RAM = ["--cache-ram", "120", "--ctx-checkpoints", "8"]


def run_traffic(server: ServerProcess, prompts: list) -> dict:
    first = complete(server, prompts[0])
    for p in prompts[1:]:
        complete(server, p)
    return first


def test_restored_checkpoint_avoids_reprocessing(tmp_path):
    """Tier on: the oldest prompt's state AND its checkpoints were spilled; the repeat restores both and re-processes only the
    last few tokens. The checkpoints are what makes this possible on a hybrid model (the state alone cannot be rolled back)."""
    global server
    d = tmp_path / "tier"
    server.extra_args = CACHE_RAM + ["--cache-disk-dir", str(d), "--cache-disk-min-tokens", "1"]
    server.start()
    prompts = make_prompts(6)
    first = run_traffic(server, prompts)
    m = metrics(server)

    # the behaviour first (this is what a build without the feature gets wrong), then the counters that explain it
    body = complete(server, prompts[0])
    prompt_n, cache_n = timings(body)
    n_total = prompt_n + cache_n
    print(f"RESTORE RESULT tier on: prompt_n={prompt_n} cache_n={cache_n} of {n_total}; "
          f"ckpt_spill={m.get('llamacpp:prompt_cache_disk_ckpt_spill_total', -1):.0f}")
    assert cache_n >= n_total * 9 // 10, f"the spilled prompt was not restored: cache_n={cache_n} of {n_total}"
    assert prompt_n <= 16
    assert body["content"] == first["content"], "same greedy answer as the first time the prompt was computed from scratch"

    assert m["llamacpp:prompt_cache_disk_spill_total"] >= 3, "nothing was spilled although the RAM cache overflowed"
    assert m["llamacpp:prompt_cache_disk_ckpt_spill_total"] >= 2 * m["llamacpp:prompt_cache_disk_spill_total"], \
        "the checkpoints of the spilled entries did not go to the tier"
    assert len(list(d.glob("pckpt-*.bin"))) > 0 and len(list(d.glob("pcache-*.bin"))) > 0
    # memory holds at most the one entry the 120 MiB limit allows, never the checkpoints of spilled ones
    assert m["llamacpp:prompt_cache_ram_bytes"] <= 120 * 1024 * 1024
    m2 = metrics(server)
    assert m2["llamacpp:prompt_cache_disk_restore_total"] >= 1
    assert m2["llamacpp:prompt_cache_disk_ckpt_restore_total"] >= 1, "the restore did not come through a tier checkpoint"
    assert m2["llamacpp:prompt_cache_disk_ckpt_read_bytes_total"] > 0
    assert m2["llamacpp:prompt_cache_disk_ckpt_restore_failed_total"] == 0
    assert m2["llamacpp:prompt_cache_disk_checksum_failed_total"] == 0


def test_control_tier_off_reprocesses():
    """Control: the same traffic without the tier loses the oldest prompt, so the repeat is a full re-process."""
    global server
    server.extra_args = CACHE_RAM
    server.start()
    prompts = make_prompts(6)
    first = run_traffic(server, prompts)
    prompt_n, cache_n = timings(complete(server, prompts[0]))
    n_total = prompt_n + cache_n
    print(f"RESTORE RESULT tier off (control): prompt_n={prompt_n} cache_n={cache_n} of {n_total}")
    assert cache_n < n_total // 4, f"without the tier the evicted prompt must be recomputed, cache_n={cache_n} of {n_total}"
    assert not any(k.startswith("llamacpp:prompt_cache_disk") for k in metrics(server))
    assert first["content"]


def test_corrupted_checkpoint_blob_is_recomputed_not_used(tmp_path):
    """Only the checkpoint blobs are damaged: the entry's main state still restores, the damaged checkpoint is refused by the
    tier's checksum, nothing wrong reaches the model, the prompt is re-processed and the answer is the same as before."""
    global server
    d = tmp_path / "tier"
    server.extra_args = CACHE_RAM + ["--cache-disk-dir", str(d), "--cache-disk-min-tokens", "1"]
    server.start()
    prompts = make_prompts(6)
    first = run_traffic(server, prompts)
    blobs = sorted(d.glob("pckpt-*.bin"))
    assert blobs, "no checkpoint blob was written"
    mains = {f: f.read_bytes() for f in d.glob("pcache-*.bin")}
    for f in blobs:                       # damage every checkpoint blob, so whichever one is chosen is bad
        raw = bytearray(f.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        f.write_bytes(bytes(raw))
    assert all(f.read_bytes() == v for f, v in mains.items()), "the control: entry blobs must be untouched"

    body = complete(server, prompts[0])
    prompt_n, cache_n = timings(body)
    n_total = prompt_n + cache_n
    print(f"RESTORE RESULT corrupted checkpoint blobs: prompt_n={prompt_n} cache_n={cache_n} of {n_total}")
    assert cache_n < n_total // 4, f"a corrupted checkpoint was used: cache_n={cache_n} of {n_total}"
    m = metrics(server)
    assert m["llamacpp:prompt_cache_disk_checksum_failed_total"] >= 1
    assert m["llamacpp:prompt_cache_disk_ckpt_restore_failed_total"] >= 1
    assert m["llamacpp:prompt_cache_disk_restore_total"] >= 1, "the entry's main state is intact and was restored"
    assert body["content"] == first["content"], "the recompute must give the answer the original computation gave"


# ---- a slot's own live checkpoints: --ctx-ckpt-spill-dir together with --cache-disk-dir -------------------------------------
# With the tier on, a checkpoint the slot moves out of memory (--ctx-ckpt-live-n) goes to the tier instead of the slot's own
# ckpt-s*.bin file, so it shares the tier's size cap, SSD write budget and checksum. A 623-token prompt makes two checkpoints
# (at ~107 and ~619); with --ctx-ckpt-live-n 1 the first one is moved out when the second is created. A request that shares only
# the first ~125 tokens with that prompt can only be served from the FIRST checkpoint, i.e. from the spilled one.

def shared_prefix_variant(prompts: list, n_shared_words: int = 120) -> str:
    head = prompts[0].split(" ")[: 2 + n_shared_words]            # "story", "0:", then the shared words
    tail = prompts[1].split(" ")[2: 2 + 480]                      # words of another prompt: the texts diverge here
    return " ".join(head + tail)


SLOT_ARGS = ["--cache-ram", "4000", "--ctx-checkpoints", "8", "--ctx-ckpt-live-n", "1"]


def test_slot_checkpoint_goes_to_the_tier_and_is_restored(tmp_path):
    global server
    spill, tier = tmp_path / "slot", tmp_path / "tier"
    server.extra_args = SLOT_ARGS + ["--ctx-ckpt-spill-dir", str(spill), "--cache-disk-dir", str(tier), "--cache-disk-min-tokens", "1"]
    server.start()
    prompts = make_prompts(2)
    complete(server, prompts[0])
    m = metrics(server)
    assert m["llamacpp:prompt_cache_disk_ckpt_slot_spill_total"] >= 1, "the slot's older checkpoint was not moved to the tier"
    assert len(list(tier.glob("pckpt-*.bin"))) >= 1
    assert len(list(spill.glob("ckpt-s*.bin"))) == 0, "with the tier on, no per-checkpoint file may be written"

    prompt_n, cache_n = timings(complete(server, shared_prefix_variant(prompts)))
    print(f"SLOT CKPT RESULT tier: prompt_n={prompt_n} cache_n={cache_n}")
    assert cache_n >= 100, f"the spilled first checkpoint was not used: cache_n={cache_n}"
    m2 = metrics(server)
    assert m2["llamacpp:prompt_cache_disk_ckpt_restore_total"] >= 1
    assert m2["llamacpp:prompt_cache_disk_ckpt_restore_failed_total"] == 0


def test_slot_checkpoint_blob_corruption_is_recomputed(tmp_path):
    global server
    spill, tier = tmp_path / "slot", tmp_path / "tier"
    server.extra_args = SLOT_ARGS + ["--ctx-ckpt-spill-dir", str(spill), "--cache-disk-dir", str(tier), "--cache-disk-min-tokens", "1"]
    server.start()
    prompts = make_prompts(2)
    complete(server, prompts[0])
    blobs = list(tier.glob("pckpt-*.bin"))
    assert blobs
    for f in blobs:
        raw = bytearray(f.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        f.write_bytes(bytes(raw))
    prompt_n, cache_n = timings(complete(server, shared_prefix_variant(prompts)))
    print(f"SLOT CKPT RESULT corrupted: prompt_n={prompt_n} cache_n={cache_n}")
    assert cache_n < 20, f"a corrupted slot checkpoint was used: cache_n={cache_n}"
    m = metrics(server)
    assert m["llamacpp:prompt_cache_disk_checksum_failed_total"] >= 1
    assert m["llamacpp:prompt_cache_disk_ckpt_restore_failed_total"] >= 1


def test_control_slot_checkpoint_files_without_the_tier(tmp_path):
    """Control: without --cache-disk-dir the same flags write the per-checkpoint file, as before, and it restores too."""
    global server
    spill = tmp_path / "slot"
    server.extra_args = SLOT_ARGS + ["--ctx-ckpt-spill-dir", str(spill)]
    server.start()
    prompts = make_prompts(2)
    complete(server, prompts[0])
    assert len(list(spill.glob("ckpt-s*.bin"))) >= 1, "the slot's own file spill must still be used when the tier is off"
    prompt_n, cache_n = timings(complete(server, shared_prefix_variant(prompts)))
    print(f"SLOT CKPT RESULT files (control): prompt_n={prompt_n} cache_n={cache_n}")
    assert cache_n >= 100
    assert not any(k.startswith("llamacpp:prompt_cache_disk") for k in metrics(server))


# ---- restore versus recompute (--cache-disk-min-speedup), row S69-TIERED-RESTORE-VS-RECOMPUTE-POLICY-2026-10-05-01 -------------
# MEASURED on this host (tests/bench-tier-restore.cpp): restoring a 23 MB state of this model takes ~20 ms, recomputing 623 tokens
# ~900 ms, so with the default margin (1.0) the rule never trips here and the restore tests above hold. A margin no host can meet
# turns the rule into "recompute is always quicker": nothing past the first spill is written, a spilled match is not restored.

def test_rule_exports_the_measured_rates(tmp_path):
    global server
    server.extra_args = CACHE_RAM + ["--cache-disk-dir", str(tmp_path / "tier"), "--cache-disk-min-tokens", "1"]
    server.start()
    prompts = make_prompts(6)
    run_traffic(server, prompts)
    complete(server, prompts[0])
    m = metrics(server)
    pf = m["llamacpp:prompt_cache_disk_prefill_tokens_per_second"]
    rs = m["llamacpp:prompt_cache_disk_restore_bytes_per_second"]
    print(f"RULE RATES: prefill {pf:.0f} tok/s, restore {rs / 1e6:.0f} MB/s, min_speedup {m['llamacpp:prompt_cache_disk_min_speedup']}")
    assert 50 < pf < 100000, "the server must have measured its own prompt processing speed"
    assert 5e6 < rs < 5e10, "the server must have measured a restore speed"
    assert m["llamacpp:prompt_cache_disk_min_speedup"] == 1.0
    assert m["llamacpp:prompt_cache_disk_dropped_slow_total"] == 0, "control: with the default margin nothing may be judged too slow here"
    assert m["llamacpp:prompt_cache_disk_restore_skipped_total"] == 0


def test_impossible_margin_stops_spilling_and_restoring(tmp_path):
    global server
    d = tmp_path / "tier"
    server.extra_args = CACHE_RAM + ["--cache-disk-dir", str(d), "--cache-disk-min-tokens", "1", "--cache-disk-min-speedup", "100000000"]
    server.start()
    prompts = make_prompts(6)
    run_traffic(server, prompts)
    m = metrics(server)
    # the first spill happens before any rate is known (the rule has no opinion); after it the tier's own write speed is known
    assert m["llamacpp:prompt_cache_disk_dropped_slow_total"] >= 1, "later evictions must be judged too slow to spill"
    assert m["llamacpp:prompt_cache_disk_spill_total"] <= 1
    prompt_n, cache_n = timings(complete(server, prompts[0]))
    n_total = prompt_n + cache_n
    print(f"RULE RESULT impossible margin: prompt_n={prompt_n} cache_n={cache_n} dropped_slow={m['llamacpp:prompt_cache_disk_dropped_slow_total']:.0f}")
    m2 = metrics(server)
    # whichever way the oldest prompt went (dropped as too slow, or matched and skipped), it was recomputed
    assert cache_n < n_total // 4
    assert m2["llamacpp:prompt_cache_disk_restore_total"] == 0
