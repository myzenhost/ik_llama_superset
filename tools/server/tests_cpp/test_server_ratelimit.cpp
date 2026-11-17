// Standalone test for server-ratelimit.h. Build: g++ -std=c++17 -I.. test_server_ratelimit.cpp -o t && ./t
// Injected clock, no server, no model. Each block carries an opposite-answer control.
#include "server-ratelimit.h"

#include <cstdio>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

// stand-in for string_parse_duration_ms: "<n>s", "<n>ms", "<n>m", bare number = seconds
static bool dur(const std::string & s, int64_t & out) {
    if (s.empty()) return false;
    size_t i = 0;
    while (i < s.size() && std::isdigit((unsigned char) s[i])) i++;
    if (i == 0) return false;
    const int64_t n = std::strtoll(s.substr(0, i).c_str(), nullptr, 10);
    const std::string u = s.substr(i);
    if (u.empty() || u == "s") { out = n * 1000; return true; }
    if (u == "ms")             { out = n;        return true; }
    if (u == "m")              { out = n * 60000; return true; }
    return false;
}

int main() {
    // ---- off: nothing limited, nothing kept ----------------------------------------------------------------------
    {
        server_rate_limit rl;
        for (int i = 0; i < 1000; i++) {
            CHECK(rl.hit("a:1.2.3.4", 0).allowed);
        }
        CHECK(rl.n_clients() == 0);
        rl.calls = 2;               // calls without a period is still off
        CHECK(!rl.enabled());
        CHECK(rl.hit("a:1.2.3.4", 0).allowed);
    }

    // ---- N per window, the N+1th rejected with the time left -----------------------------------------------------
    {
        server_rate_limit rl;
        rl.calls = 2;
        rl.period_ms = 1000;
        auto v1 = rl.hit("a:A", 0);
        CHECK(v1.allowed && v1.remaining == 1 && v1.limit == 2 && v1.reset_ms == 1000);
        auto v2 = rl.hit("a:A", 100);
        CHECK(v2.allowed && v2.remaining == 0 && v2.reset_ms == 900);
        auto v3 = rl.hit("a:A", 300);
        CHECK(!v3.allowed);                      // the third within the window
        CHECK(v3.remaining == 0);
        CHECK(v3.reset_ms == 700);               // window anchored at the FIRST request (t=0), not at the rejection
        // control: another client is unaffected
        auto o1 = rl.hit("a:B", 300);
        CHECK(o1.allowed && o1.remaining == 1);
        // a rejected request does not move the window
        CHECK(!rl.hit("a:A", 999).allowed);
        // the window ends: allowed again, new window from now
        auto v4 = rl.hit("a:A", 1000);
        CHECK(v4.allowed && v4.remaining == 1 && v4.reset_ms == 1000);
        // a clock that goes backwards starts a new window
        auto v5 = rl.hit("a:A", 10);
        CHECK(v5.allowed);
    }

    // ---- sweep: ended windows go, running ones stay-------------------------------------------------------------------
    {
        server_rate_limit rl;
        rl.calls = 1;
        rl.period_ms = 1000;
        rl.sweep_above = 3;
        for (int i = 0; i < 4; i++) {
            rl.hit("a:" + std::to_string(i), 0);
        }
        CHECK(rl.n_clients() == 4);
        rl.hit("a:live", 1500);                  // > sweep_above: the 4 ended windows go, the new one stays
        CHECK(rl.n_clients() == 1);
        // control: a sweep keeps a window that is still running
        rl.hit("a:x", 1600); rl.hit("a:y", 1600); rl.hit("a:z", 1600);
        rl.hit("a:w", 1700);
        CHECK(!rl.hit("a:live", 1800).allowed);  // its window (1500..2500) survived the sweep
    }

    // ---- client keys ---------------------------------------------------------------------------------------------
    CHECK(server_rate_limit_client("addr", "1.2.3.4", "sk-1") == "a:1.2.3.4");
    CHECK(server_rate_limit_client("key",  "1.2.3.4", "")     == "a:1.2.3.4");  // no key presented: address
    {
        const std::string k1 = server_rate_limit_client("key", "1.2.3.4", "sk-1");
        const std::string k2 = server_rate_limit_client("key", "1.2.3.4", "sk-2");
        const std::string k3 = server_rate_limit_client("key", "5.6.7.8", "sk-1");
        CHECK(k1 != k2);                         // two keys behind one address are two clients
        CHECK(k1 == k3);                         // one key from two addresses is one client
        CHECK(k1.find("sk-1") == std::string::npos); // the table holds no raw key
    }

    // ---- parse ---------------------------------------------------------------------------------------------------
    {
        int64_t c = 0, p = 0;
        CHECK(server_rate_limit::parse("2/1s", c, p, dur) && c == 2 && p == 1000);
        CHECK(server_rate_limit::parse("60/minute", c, p, dur) && c == 60 && p == 60000);
        CHECK(server_rate_limit::parse("100 / 10 minutes", c, p, dur) && c == 100 && p == 600000);
        CHECK(server_rate_limit::parse("5/second", c, p, dur) && c == 5 && p == 1000);
        CHECK(server_rate_limit::parse("1000/day", c, p, dur) && c == 1000 && p == 86400000LL);
        CHECK(server_rate_limit::parse("3/500ms", c, p, dur) && c == 3 && p == 500);
        CHECK(server_rate_limit::parse("3/30", c, p, dur) && c == 3 && p == 30000);  // bare number = seconds
        CHECK(server_rate_limit::parse("3/2m", c, p, dur) && c == 3 && p == 120000);
        // malformed specs
        CHECK(!server_rate_limit::parse("0/1s", c, p, dur));
        CHECK(!server_rate_limit::parse("-1/1s", c, p, dur));
        CHECK(!server_rate_limit::parse("2", c, p, dur));
        CHECK(!server_rate_limit::parse("2/", c, p, dur));
        CHECK(!server_rate_limit::parse("/1s", c, p, dur));
        CHECK(!server_rate_limit::parse("2/fortnight", c, p, dur));
        CHECK(!server_rate_limit::parse("2/0", c, p, dur));
        CHECK(!server_rate_limit::parse("x/1s", c, p, dur));
    }

    if (fails == 0) {
        std::printf("test_server_ratelimit: all checks passed\n");
        return 0;
    }
    std::printf("test_server_ratelimit: %d check(s) failed\n", fails);
    return 1;
}
