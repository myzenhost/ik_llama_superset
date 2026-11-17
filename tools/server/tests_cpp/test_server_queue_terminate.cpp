// server_queue::terminate() issued BEFORE start_loop() must not be lost, model-free.
// Row S69-ENGREV-JAN-QUEUE-TERMINATE-BEFORE-LOOP-LOST-2026-10-27-01. Jan (Apache-2.0) guards the same race in its server shim and
// tests it (engine_smoke.rs); source read for the scenario only, no code copied.
// Before the fix start_loop() began with `running = true`, overwriting the terminate(); the loop then blocked forever (the
// SIGTERM handler in server.cpp calls terminate() on a server that is still loading its model, before start_loop()).
// GATE-CLAIM-OK: unit test, no resource claims.
#include "server-task.h"
#include "server-queue.h"

#include <chrono>
#include <cstdio>
#include <future>
#include <string>
#include <thread>

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    n_fail += ok ? 0 : 1;
}

// run q.start_loop() on its own thread; returns true when it came back within `ms`. On timeout the loop is released with a
// second terminate() so the test process can still exit (and report FAIL instead of hanging).
static bool loop_returns(server_queue & q, int ms, bool terminate_after_start) {
    std::promise<void> done;
    auto fut = done.get_future();
    std::thread t([&]() {
        q.start_loop();
        done.set_value();
    });
    if (terminate_after_start) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        q.terminate();
    }
    const bool ok = fut.wait_for(std::chrono::milliseconds(ms)) == std::future_status::ready;
    if (!ok) {
        q.terminate(); // release the hung loop
    }
    t.join();
    return ok;
}

int main() {
    // 1. the bug: terminate() first, start_loop() second
    {
        server_queue q;
        q.on_update_slots([]() {});
        q.terminate();
        check(loop_returns(q, 3000, false), "terminate() before start_loop(): the loop returns at once");
    }
    // 2. control: terminate() while the loop runs still ends it (the working path must stay working)
    {
        server_queue q;
        q.on_update_slots([]() {});
        check(loop_returns(q, 3000, true), "terminate() after start_loop(): the loop returns");
    }
    // 3. terminate() is consumed by the start_loop() it stopped: a later loop on the same queue is not stopped by the old request
    {
        server_queue q;
        q.on_update_slots([]() {});
        q.terminate();
        const bool first = loop_returns(q, 3000, false);
        const bool second_runs_until_terminated = loop_returns(q, 400, false) == false; // times out = it kept running; helper then releases it
        check(first && second_runs_until_terminated, "a pending terminate() stops exactly one start_loop(); the next loop runs until terminated");
    }
    printf("%s\n", n_fail ? "FAILED" : "ALL PASSED");
    return n_fail ? 1 : 0;
}
