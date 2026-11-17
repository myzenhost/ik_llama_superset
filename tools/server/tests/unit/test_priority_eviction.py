import threading
import time

import pytest
from utils import ServerPreset, use_local_stories260k

# Priority eviction on a full queue (row S69-TFSERVING-PICK-PRIORITY-EVICTION-ON-FULL-QUEUE-2026-10-15-01).
# With --max-queued-requests 2 the server holds two requests in flight: one running and one waiting for the slot. A third arrival is
# refused 503 "server busy". With --evict-low-priority, an arrival whose `priority` is STRICTLY more important (lower value) than the
# waiting request takes its place: the waiting one is answered 503 "evicted", the arrival is admitted and runs next. Equal or worse
# priority never evicts, and a request that already has its slot is never evicted.
# Contract read from tensorflow shared_batch_scheduler.h:452-503 (PriorityTaskQueue::AddTask): victim = lowest criticality, newest
# within the class; evict only if the new entry sorts strictly before the victim.

HOLD_MS = 2500  # how long A holds the slot; stories260K decodes at 4-20 ms/token on this CPU depending on host load, so the token count
                # is derived from a calibration request on the same server

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_ctx = 4096
    server.n_slots = 1
    server.n_batch = 64
    server.server_slots = True
    server.server_port = 18434  # NOT 8081: that port is the live serving instance on this host


def _hold_tokens():
    r = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 40, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}, timeout=120)
    t = r.body["timings"]
    return int(min(3000, max(20, HOLD_MS / max(1.0, t["predicted_ms"] / max(1, t["predicted_n"])))))


def _post(box, key, priority, n_predict):
    box[key] = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": n_predict, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False,
        "priority": priority}, timeout=300)


def _spawn(box, key, priority, n_predict):
    t = threading.Thread(target=_post, args=(box, key, priority, n_predict))
    t.start()
    return t


def _admission():
    return server.make_request("GET", "/props").body["admission"]


def _wait(cond, what, seconds=20):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for " + what)


def _fill_the_queue(box):
    """A (priority 0) runs, B (priority 5) waits: two in flight = the ceiling."""
    box["n_hold"] = _hold_tokens()
    ta = _spawn(box, "A", 0, box["n_hold"])
    _wait(lambda: any(s.get("is_processing") for s in server.make_request("GET", "/slots").body), "A to hold the slot")
    tb = _spawn(box, "B", 5, 5)
    _wait(lambda: _admission()["in_flight_requests"] == 2, "B to be admitted behind A")
    return ta, tb


def test_higher_priority_arrival_evicts_the_waiting_low_priority_request():
    global server
    server.extra_args = ["--max-queued-requests", "2", "--evict-low-priority"]
    server.start()
    box = {}
    ta, tb = _fill_the_queue(box)
    c = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 5, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False, "priority": 1}, timeout=300)
    tb.join()
    ta.join()
    # C took B's place and ran after A
    assert c.status_code == 200 and c.body["tokens_predicted"] == 5, (c.status_code, str(c.body)[:200])
    assert box["A"].status_code == 200 and box["A"].body["tokens_predicted"] == box["n_hold"]
    assert box["B"].status_code == 503, (box["B"].status_code, str(box["B"].body)[:200])
    assert "evicted" in box["B"].body["error"]["message"], box["B"].body
    a = _admission()
    assert a["evicted"] == 1, a
    # no ticket leaked: the evicted request's ticket and the arrival's both went back
    assert a["in_flight_requests"] == 0 and a["prefill_tokens"] == 0, a


def test_equal_or_worse_priority_never_evicts():
    global server
    server.extra_args = ["--max-queued-requests", "2", "--evict-low-priority"]
    server.start()
    box = {}
    ta, tb = _fill_the_queue(box)
    equal = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 2, "temperature": 0.0, "priority": 5}, timeout=60)
    worse = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 2, "temperature": 0.0, "priority": 9}, timeout=60)
    ta.join()
    tb.join()
    assert equal.status_code == 503 and "server busy" in equal.body["error"]["message"], (equal.status_code, str(equal.body)[:200])
    assert worse.status_code == 503 and "server busy" in worse.body["error"]["message"], (worse.status_code, str(worse.body)[:200])
    assert box["B"].status_code == 200, "the waiting request must survive an equal or worse arrival"
    assert _admission()["evicted"] == 0


def test_a_running_request_is_never_the_victim():
    global server
    server.extra_args = ["--max-queued-requests", "1", "--evict-low-priority"]
    server.start()
    box = {}
    box["n_hold"] = _hold_tokens()
    ta = _spawn(box, "A", 5, box["n_hold"])  # low priority, but it HAS the slot
    _wait(lambda: any(s.get("is_processing") for s in server.make_request("GET", "/slots").body), "A to hold the slot")
    urgent = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 2, "temperature": 0.0, "priority": -10}, timeout=60)
    ta.join()
    assert urgent.status_code == 503 and "server busy" in urgent.body["error"]["message"], (urgent.status_code, str(urgent.body)[:200])
    assert box["A"].status_code == 200 and box["A"].body["tokens_predicted"] == box["n_hold"]


def test_control_without_the_flag_refuses_the_arrival():
    global server
    server.extra_args = ["--max-queued-requests", "2"]
    server.start()
    box = {}
    ta, tb = _fill_the_queue(box)
    c = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 5, "temperature": 0.0, "priority": 1}, timeout=60)
    ta.join()
    tb.join()
    assert c.status_code == 503 and "server busy" in c.body["error"]["message"], (c.status_code, str(c.body)[:200])
    assert box["B"].status_code == 200, "without --evict-low-priority the waiting request keeps its place"
