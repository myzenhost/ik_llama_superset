"""Regression test for the harness defect that made the radix-cache test module pollute every later test in the session.

Row S69-ADD-TGI-ALTERNATIVES-LLGUIDANCE-PAGEDKV-RADIX-SYNTHID-2026-09-27-01 (wave-3 merge check 2026-09-28): radix 3/4, watermark 1/7,
cross_slot 0/3 in ONE pytest session, while the same watermark tests passed 7/7 alone. Root cause (reproduced 2026-10-01, see
tmp/research/findings/S69_TESTS_RESIDUAL_2026-10-01.md): a test called ServerProcess.start() twice on one object. The second
call overwrote self.process, teardown stopped only the second server, and the first kept listening. Windows lets two servers bind one
port, so the leftover answered later tests ("watermark requested but the server has no --watermark-key", "c ran on slot 0, expected 2").

Two fixes are in the tree: ServerProcess.start() stops a still-running previous process first (utils.py), and the radix helper
calls server.stop() before starting the next server. This test pins the harness half so it cannot be removed unnoticed: it fails with
the guard neutralised (first process still running after the second start) and passes with it.
"""
from utils import ServerPreset, use_local_stories260k


def test_second_start_stops_the_first_process():
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 1
    server.n_ctx = 512
    server.n_predict = 4

    server.start()
    first = server.process
    assert first is not None and first.poll() is None, "the first server did not start"

    server.start()  # the second call on the same object must not leave the first one running
    second = server.process
    try:
        assert second is not None and second.pid != first.pid
        assert first.poll() is not None, (
            f"first server pid={first.pid} is still running after a second start() on the same ServerProcess: "
            "it keeps the port and answers later tests (the 2026-09-28 session pollution)")
    finally:
        server.stop()
        if first.poll() is None:  # never leave the leaked server behind, even when the assertion above fires
            first.kill()
            first.wait(timeout=10)
