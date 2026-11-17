# A SWA state that holds the window the next evaluated token needs must be reused as is
# (row S69-LLAMACPP-PICK-SWA-REUSE-THRESHOLD-OFF-BY-TWO-2026-10-18-01).
#
# The first evaluated token at pos_eval attends to positions > pos_eval - n_swa, so a SWA state whose pos_min is
# pos_eval - n_swa + 1 holds the whole window. The old reuse threshold asked for two more cells, and a saved SWA state
# keeps only the cells inside its window, so a state read back from a slot file or from the RAM prompt cache was always
# refused and the slot rolled back to a checkpoint (or reprocessed the prompt).
# Behaviour and test shape from upstream llama.cpp 61ad80bf (#30580, MIT), read whole; rewritten for this harness
# (extra_args instead of an override_kv attribute, cross-slot publish off so the measured slot is the restored one).
import pytest
from utils import ServerPreset


@pytest.fixture
def swa_window_server(tmp_path):
    # a SWA window smaller than the prompt, so a saved state holds only the cells inside the window
    swa = ServerPreset.tinygemma3()
    swa.extra_args = ["--override-kv", "gemma3.attention.sliding_window=int:64"]
    swa.slot_save_path = str(tmp_path)
    swa.temperature = 0.0
    swa.n_predict = 8
    swa.no_cache_publish_on_prefill = True
    return swa


SWA_PROMPT = "The quick brown fox jumps over the lazy dog. " * 30


def tokenize(server, text):
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    return res.body["tokens"]


# the first turn and the exact token continuation of it, so the next turn extends the saved state
def swa_turns(server, id_slot):
    res = server.make_request("POST", "/completion", data={
        "prompt": tokenize(server, SWA_PROMPT),
        "id_slot": id_slot,
        "cache_prompt": True,
        "return_tokens": True,
    })
    assert res.status_code == 200
    return tokenize(server, SWA_PROMPT) + res.body["tokens"] + tokenize(server, " The end.")


def swa_next(server, id_slot, prompt):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": id_slot,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body["timings"]


# a restored SWA state holds the window the next token needs, so it is reused like the live slot's state
def test_slot_restore_swa_window(swa_window_server):
    server = swa_window_server
    server.cache_ram = 0
    server.start()

    prompt = {id_slot: swa_turns(server, id_slot) for id_slot in [0, 1]}
    assert prompt[0] == prompt[1]
    assert len(prompt[0]) > 2 * 64  # the prompt is longer than the window, else the test proves nothing

    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "swa_slot0.bin"})
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "swa_slot0.bin"})
    assert res.status_code == 200

    timings = {id_slot: swa_next(server, id_slot, prompt[id_slot]) for id_slot in [0, 1]}
    print("timings restored slot 0:", timings[0], "live slot 1:", timings[1])

    # control: the live slot reuses its state (cache_n > 0), so equality below is not two full reprocesses
    assert timings[1]["cache_n"] > 0
    assert timings[0]["cache_n"] == timings[1]["cache_n"]
    assert timings[0]["prompt_n"] == timings[1]["prompt_n"]


# a SWA state loaded back from the RAM prompt cache is reused like the state that never left the slot
def test_prompt_cache_swa_window(swa_window_server):
    server = swa_window_server
    server.n_slots = 1

    def run(other_prompt_first):
        server.start()
        prompt = swa_turns(server, 0)
        if other_prompt_first:
            res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time " * 60, "cache_prompt": True})
            assert res.status_code == 200
        timings = swa_next(server, 0, prompt)
        server.stop()
        return timings

    direct = run(False)
    cached = run(True)
    print("timings direct:", direct, "via RAM prompt cache:", cached)

    assert direct["cache_n"] > 0
    assert cached["cache_n"] == direct["cache_n"]
    assert cached["prompt_n"] == direct["prompt_n"]


# ---- the two tests above cannot tell the thresholds apart on this fork -------------------------------------------------
# Row S69-FORK-SWA-REUSE-E2E-REACH-VIA-SWA-COMPRESS-2026-10-19-01. On the legacy KV path a slot's pos_min is its first
# cell (src/llama.cpp llama_kv_cache_seq_pos_min), which stays 0 unless the SWA layers are compacted, so the old and the
# new threshold both accept. With --swa-compress (gemma3 is on the compaction allowlist, src/llama-model.h
# supports_swa_compress) the SWA rows really hold only the window, and pos_min is raised to the window base
# (pos_base_swa).
#
# The arithmetic, with window W = 64, -fa off (pad 32) and -b 256 -ub 256:
#   compacted rows C = W + max(n_ubatch, 256) = 320 (src/llama-hparams.h llama_swa_compact_rows)
#   turn 1 (> 320 tokens) is decoded in two calls, [0, 256) and [256, len). The second call does not fit (256 + n > 320),
#   so the window rolls first and keeps the last W positions: pos_base_swa = 256 - 64 = 192
#   (src/llama.cpp llama_kv_cache_compact_swa), and the window can be rewound to any p0 >= 192 + 64 = 256.
#   turn 2 shares exactly the first 256 tokens: pos_eval = 256, pos_min = 192 = pos_eval - W.
#   old threshold pos_eval - W = 192: pos_min is not below it, the state is refused and the prompt reprocessed (cache_n 0)
#   new threshold pos_eval - W + 2 = 194: the state is reused (cache_n 256). Upstream llama.cpp 61ad80bf (#30580, MIT).
# Control: a turn 2 that shares 300 tokens is reused under both thresholds, so a failure at 256 is the threshold and not
# a compacted cache that cannot be reused at all.
SWA_W = 64
SWA_COMPRESS_PROMPT = "The quick brown fox jumps over the lazy dog. " * 40
SWA_COMPRESS_TAIL = " Zebras painted the moon green while the owls sang softly to the sleeping hills."


def swa_compress_server(log_path):
    server = ServerPreset.tinygemma3()
    server.extra_args = [
        "--override-kv", "gemma3.attention.sliding_window=int:%d" % SWA_W,
        "--swa-compress",
        "--ctx-checkpoints", "0",  # no checkpoint to fall back to: a refused state means a full reprocess
    ]
    server.n_slots = 1             # the compacted window is shared by the context (one head), so one slot
    server.n_batch = 256
    server.n_ubatch = 256
    server.fa = "off"              # pad 32, so C = 320 (with FA the pad is 256 and C = 512)
    server.no_mmproj = True
    server.cache_ram = 0           # no RAM prompt cache: the measured reuse is the slot's own state
    server.no_cache_publish_on_prefill = True
    server.temperature = 0.0
    server.n_predict = 8
    server.log_path = log_path
    return server


def swa_compress_turn2(tmp_path, n_shared):
    log_path = str(tmp_path / ("swa_compress_%d.log" % n_shared))
    server = swa_compress_server(log_path)
    server.start()
    t1 = tokenize(server, SWA_COMPRESS_PROMPT)
    # two decode calls, the second one larger than C - live = 64 so it rolls the window; and no roll during turn 1's
    # generation (64 + (len - 256) + 8 <= 320)
    assert 256 + SWA_W < len(t1) <= 512 - 8, len(t1)
    assert n_shared < len(t1)
    res = server.make_request("POST", "/completion", data={"prompt": t1, "cache_prompt": True})
    assert res.status_code == 200

    t2 = t1[:n_shared] + tokenize(server, SWA_COMPRESS_TAIL)
    assert t2[n_shared] != t1[n_shared]  # the shared prefix is exactly n_shared tokens
    res = server.make_request("POST", "/completion", data={"prompt": t2, "cache_prompt": True, "n_probs": 3})
    assert res.status_code == 200
    reused = res.body
    # reference: the same prompt reprocessed from scratch
    res = server.make_request("POST", "/completion", data={"prompt": t2, "cache_prompt": False, "n_probs": 3})
    assert res.status_code == 200
    fresh = res.body
    server.stop()
    with open(log_path, encoding="utf-8", errors="replace") as f:
        log = f.read()
    print("n_shared %d: reused timings %s content %r | fresh timings %s content %r" % (
        n_shared, reused["timings"], reused["content"], fresh["timings"], fresh["content"]))
    return reused, fresh, log


@pytest.mark.parametrize("n_shared", [256, 300])
def test_swa_compress_window_exact_state_is_reused(tmp_path, n_shared):
    reused, fresh, log = swa_compress_turn2(tmp_path, n_shared)
    # reach: compaction engaged (src/llama.cpp llama_kv_cache_init), so pos_min is the window base and not 0
    assert "--swa-compress engaged" in log
    assert fresh["timings"]["cache_n"] == 0
    # 256: refused by the old threshold (cache_n 0), reused by the new one; 300: reused by both (control)
    assert reused["timings"]["cache_n"] == n_shared
    # the reused state gives the same answer as reprocessing the prompt
    assert reused["content"] == fresh["content"]


# Sharing 255 tokens: pos_eval = 255 and pos_min = 192 = pos_eval - W + 1, which the new threshold (193) accepts, but the
# compacted window can only be rewound to p0 >= pos_base_swa + W = 256 (src/llama.cpp llama_kv_cache_seq_rm), so the
# server's seq_rm(255) was refused and common_context_seq_rm aborted the server (MEASURED 2026-10-19 on af53c08fb:
# ConnectionResetError, the probe at 254/256/257/300 was fine). The server now treats such a state as unusable
# (tools/server/server-swa-reuse.h server_swa_reuse_rewind_ok) and reprocesses the prompt.
def test_swa_compress_state_below_rewind_floor_is_reprocessed(tmp_path):
    reused, fresh, log = swa_compress_turn2(tmp_path, 255)
    assert "--swa-compress engaged" in log
    assert "cannot be rewound" in log
    assert fresh["timings"]["cache_n"] == 0
    assert reused["timings"]["cache_n"] == 0
    assert reused["content"] == fresh["content"]
