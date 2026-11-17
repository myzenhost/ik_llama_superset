// Standalone test for server-watchdog.h (row S69-ENGREV-SGLANG-FORWARD-WATCHDOG-2026-10-27-01).
// Build: g++ -std=c++17 -I.. test_server_watchdog.cpp -o t && ./t     No model, injected clock, no sleeping.
// PRIOR-ART-NONE: new feature test; nothing is switched off, the zero-timeout case only checks the unconfigured default stays quiet.
#include "server-watchdog.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

using V = server_fwd_watchdog::verdict;

int main() {
    // soft fires once per stall episode, hard after it, a beat closes the episode
    {
        server_fwd_watchdog w;
        w.configure(2000, 6000, 5000);
        CHECK(w.enabled());
        w.begin(0);
        CHECK(w.poll(1999) == V::ok);              // control: just under the soft timeout
        CHECK(w.poll(2000) == V::soft_fire);
        CHECK(w.stalled());
        CHECK(w.poll(2500) == V::ok);              // once per episode, not once per poll
        CHECK(w.poll(6000) == V::hard_fire);
        CHECK(w.poll(6500) == V::ok);
        w.beat(7000);                              // progress: episode closed
        CHECK(!w.stalled());
        CHECK(w.forward_ct() == 1);
        CHECK(w.poll(8999) == V::ok);
        CHECK(w.poll(9000) == V::soft_fire);       // a NEW stall fires again
    }
    // steady progress: beats before the timeout never fire (1000 normal steps = zero firings)
    {
        server_fwd_watchdog w;
        w.configure(500, 1000);
        w.begin(0);
        int fired = 0;
        for (int i = 1; i <= 1000; ++i) {
            w.beat((int64_t) i * 400);
            fired += w.poll((int64_t) i * 400 + 399) != V::ok;
        }
        CHECK(fired == 0);
        CHECK(w.forward_ct() == 1000);
    }
    // idle never fires, even long past the timeout
    {
        server_fwd_watchdog w;
        w.configure(100, 200);
        CHECK(w.poll(1000000) == V::ok);           // never begun
        w.begin(0);
        w.end();
        CHECK(w.poll(1000000) == V::ok);           // ended
        CHECK(w.stall_ms(1000000) == 0);
    }
    // unconfigured (zero timeouts) stays quiet; each level is independent
    {
        server_fwd_watchdog none;
        none.configure(0, 0);
        CHECK(!none.enabled());
        none.begin(0);
        CHECK(none.poll(1 << 30) == V::ok);

        server_fwd_watchdog soft_only;
        soft_only.configure(100, 0);
        soft_only.begin(0);
        CHECK(soft_only.poll(100) == V::soft_fire);
        CHECK(soft_only.poll(10000000) == V::ok);  // no hard level configured

        server_fwd_watchdog hard_only;
        hard_only.configure(0, 100);
        hard_only.begin(0);
        CHECK(hard_only.poll(100) == V::hard_fire);
    }
    // stall_ms reports the time since the last beat
    {
        server_fwd_watchdog w;
        w.configure(1000, 0);
        w.begin(10);
        w.beat(50);
        CHECK(w.stall_ms(80) == 30);
    }

    // the /health view (server_fwd_watchdog_health): nothing published -> healthy; a published watchdog reports > 0 only
    // from a fired verdict until progress resumes; withdrawn -> healthy again
    {
        server_fwd_watchdog_health h;
        CHECK(h.stalled_ms(100) == 0);                 // no watchdog published (--forward-watchdog-* off)
        server_fwd_watchdog w;
        w.configure(1000, 0);
        h.wd.store(&w);
        w.begin(0);
        CHECK(h.stalled_ms(500) == 0);                 // in flight, under the soft timeout: healthy
        CHECK(w.poll(1500) == server_fwd_watchdog::verdict::soft_fire);
        CHECK(h.stalled_ms(1500) == 1500);             // stalled: /health answers 503 with the stall length
        CHECK(h.stalled_ms(2500) == 2500);             // stays unhealthy while no step finishes (soft fires once, state persists)
        w.beat(2600);
        CHECK(h.stalled_ms(2700) == 0);                // a step finished: healthy
        CHECK(w.poll(5000) == server_fwd_watchdog::verdict::soft_fire);
        w.end();
        CHECK(h.stalled_ms(5100) == 0);                // batch ended: healthy
        h.wd.store(nullptr);
        CHECK(h.stalled_ms(9999) == 0);                // withdrawn
        CHECK(&server_fwd_watchdog_health_global() == &server_fwd_watchdog_health_global());  // one process-wide view
    }

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
