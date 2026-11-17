// Standalone test for server-retry.h. Build: g++ -std=c++17 -I.. test_server_retry.cpp -o t && ./t
// Injected clock, no server, no model. Each block carries an opposite-answer control.
#include "server-retry.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // ---- whole seconds, floor, ceiling ---------------------------------------------------------------------------
    CHECK(server_retry_after_s(0)      == 1);   // a zero wait would invite an immediate retry
    CHECK(server_retry_after_s(-5)     == 1);
    CHECK(server_retry_after_s(1)      == 1);
    CHECK(server_retry_after_s(1000)   == 1);
    CHECK(server_retry_after_s(1001)   == 2);   // ceil
    CHECK(server_retry_after_s(9500)   == 10);
    CHECK(server_retry_after_s(1000000, 1, 60) == 60);   // capped

    // ---- busy estimator ------------------------------------------------------------------------------------------
    {
        server_retry_estimator e;
        CHECK(e.busy_retry_after_s(4) == 1);             // no data: the 1 s floor, not an estimate
        e.observe_request_ms(8000);
        CHECK(e.busy_retry_after_s(4) == 2);             // 8 s / 4 slots
        CHECK(e.busy_retry_after_s(1) == 8);             // one slot: the whole request
        CHECK(e.busy_retry_after_s(0) == 8);             // a zero slot count is treated as one, not a divide by zero
        e.observe_request_ms(0);                          // clock glitches are not observations
        e.observe_request_ms(-3);
        CHECK(e.n_observations() == 1);
        e.observe_request_ms(18000);                      // EMA: 0.2 * 18000 + 0.8 * 8000 = 10000
        CHECK(e.busy_retry_after_s(1) == 10);
        // control: a much longer typical request raises the answer, a cap holds it
        server_retry_estimator slow;
        slow.observe_request_ms(600000);
        CHECK(slow.busy_retry_after_s(1) == 60);
    }

    // ---- loading: grows with the time already spent --------------------------------------------------------------
    CHECK(server_retry_estimator::loading_retry_after_s(0)      == 1);
    CHECK(server_retry_estimator::loading_retry_after_s(2000)   == 1);
    CHECK(server_retry_estimator::loading_retry_after_s(20000)  == 5);
    CHECK(server_retry_estimator::loading_retry_after_s(600000) == 10);   // capped

    // ---- load-failure cooldown -----------------------------------------------------------------------------------
    {
        server_load_cooldown c;            // base 10 s, max 300 s
        CHECK(c.cooldown_ms(1) == 10000);
        CHECK(c.cooldown_ms(2) == 20000);
        CHECK(c.cooldown_ms(3) == 40000);
        CHECK(c.cooldown_ms(6) == 300000);                // 320 s capped to 300 s
        CHECK(c.cooldown_ms(500) == 300000);              // no overflow far past the cap
        CHECK(c.cooldown_ms(0) == 0);

        CHECK(c.remaining_ms("m", 0) == 0);               // never failed: free
        CHECK(c.on_failure("m", 1000) == 10000);
        CHECK(c.remaining_ms("m", 1000) == 10000);
        CHECK(c.remaining_ms("m", 6000) == 5000);         // counts down
        CHECK(c.remaining_ms("m", 11000) == 0);           // expired
        CHECK(c.remaining_ms("other", 1000) == 0);        // per model
        CHECK(c.on_failure("m", 12000) == 20000);         // second consecutive failure doubles
        CHECK(c.n_failures("m") == 2);
        c.on_success("m");                                // success resets to the base
        CHECK(c.n_failures("m") == 0 && c.remaining_ms("m", 12001) == 0);
        CHECK(c.on_failure("m", 20000) == 10000);

        // an explicit operator action clears it
        c.clear("m");
        CHECK(c.remaining_ms("m", 20001) == 0 && c.n_failures("m") == 0);

        // base <= 0: the feature is not in use, nothing is ever refused
        server_load_cooldown off;
        off.base_ms = 0;
        CHECK(!off.enabled());
        CHECK(off.on_failure("m", 0) == 0);
        CHECK(off.remaining_ms("m", 0) == 0);
        CHECK(off.n_failures("m") == 0);

        // Retry-After from the remaining cooldown
        server_load_cooldown c2;
        c2.on_failure("x", 0);
        CHECK(server_retry_after_s(c2.remaining_ms("x", 0)) == 10);
        CHECK(server_retry_after_s(c2.remaining_ms("x", 9100)) == 1);
    }

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("test_server_retry: all checks passed\n");
    return 0;
}
