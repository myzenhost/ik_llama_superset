# priority request field (row S69-VLLM-PICK-REQUEST-PRIORITY-ADMISSION-2026-09-29-01, the queue-ordering part).
#
# SOURCE-READ: engines/vllm/vllm/v1/core/sched/request_queue.py:100-178 (waiting requests ordered by (priority, arrival_time), lower value
# first), engines/vllm/vllm/entrypoints/openai/completion/protocol.py:134-140 (the request field), engines/sglang/python/sglang/srt/managers/
# schedule_policy.py:561-572 (sort by priority then FCFS). Apache-2.0, no code copied.
#
# One slot. A long request A occupies it; B (priority 5), C (0) and D (0) queue up in that arrival order. Without priorities they would start
# A, B, C, D; with them A, C, D, B. A running request is never interrupted, so A stays first in both.
import threading
import time

import pytest

from utils import ServerPreset, use_local_stories260k  # noqa: E402

server = ServerPreset.tinyllama2()

LONG_TOKENS = 1900


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 1
    server.n_ctx = 2048
    server.n_batch = 8
    server.n_ubatch = 8
    server.server_slots = True


def _long_prompt():
    return [1] + [3 + ((i * 7) % 500) for i in range(LONG_TOKENS - 1)]


def _run(priorities):
    """A runs first; then B, C, D arrive 150 ms apart with the given priorities (None = field not sent). Returns the finish order."""
    finished = []
    lock = threading.Lock()

    def call(name, prompt, prio):
        data = {"prompt": prompt, "n_predict": 2, "temperature": 0.0, "seed": 1, "cache_prompt": False}
        if prio is not None:
            data["priority"] = prio
        res = server.make_request("POST", "/completion", data=data, timeout=180)
        assert res.status_code == 200, (name, res.status_code, res.body)
        with lock:
            finished.append(name)

    threads = [threading.Thread(target=call, args=("A", _long_prompt(), None))]
    threads[0].start()
    deadline = time.time() + 5
    while time.time() < deadline:
        slots = server.make_request("GET", "/slots").body
        if any(s.get("is_processing") for s in slots):
            break
        time.sleep(0.01)
    else:
        raise AssertionError("control: the long request must be running before the others are sent")
    for name, prio in zip("BCD", priorities):
        th = threading.Thread(target=call, args=(name, [1, 20 + ord(name), 30, 40], prio))
        th.start()
        threads.append(th)
        time.sleep(0.15)   # fixes the arrival order
    for th in threads:
        th.join(180)
    return finished


def test_without_priorities_the_queue_is_first_come_first_served():
    global server
    server.start()
    assert _run([None, None, None]) == ["A", "B", "C", "D"]


def test_lower_priority_value_starts_first_and_ties_keep_arrival_order():
    global server
    server.start()
    assert _run([5, 0, 0]) == ["A", "C", "D", "B"]


def test_priority_out_of_range_answers_400():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": [1, 2, 3], "n_predict": 1, "priority": 10 ** 9})
    assert res.status_code == 400
    assert "priority" in str(res.body)
