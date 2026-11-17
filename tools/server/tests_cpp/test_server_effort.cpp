// Standalone test for server-effort.h. Build: g++ -std=c++17 -I.. test_server_effort.cpp -o t && ./t
#include "server-effort.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // ---- the default table: koboldcpp's convention -------------------------------------------------------------------
    {
        server_effort_table t = server_effort_table::preset_default();
        CHECK(t.budget("minimal", 400) == 40);
        CHECK(t.budget("low",     400) == 100);
        CHECK(t.budget("medium",  400) == 200);
        CHECK(t.budget("high",    400) == 300);
        CHECK(t.budget("LOW",     400) == 100);        // case-insensitive
        CHECK(t.budget("none",    400) == -1);         // "none" keeps its own meaning (thinking off) elsewhere
        CHECK(t.budget("extreme", 400) == -1);         // unknown level: unrestricted
        CHECK(t.budget("",        400) == -1);
        CHECK(t.budget("low",      -1) == -1);         // no finite limit: nothing to take a fraction of
        CHECK(t.budget("low",       0) == -1);
        // floor, and at least one token for a mapped level
        CHECK(t.budget("low", 7)       == 1);          // floor(1.75) = 1
        CHECK(t.budget("minimal", 5)   == 1);          // floor(0.5) = 0, raised to 1: a zero budget means "no thinking"
        CHECK(t.budget("high", 401)    == 300);        // floor(300.75)
    }

    // ---- empty table: the feature is not in use -----------------------------------------------------------------------
    {
        server_effort_table t;
        CHECK(t.empty());
        CHECK(t.budget("low", 400) == -1);
    }

    // ---- parsing ------------------------------------------------------------------------------------------------------
    {
        server_effort_table t;
        std::string err;
        CHECK(t.parse("default", err) && t.fraction.size() == 4);
        CHECK(t.parse("low=0.2, HIGH=90%", err));
        CHECK(t.budget("low", 100) == 20);
        CHECK(t.budget("high", 100) == 90);
        CHECK(t.budget("medium", 100) == -1);          // only what the spec lists

        // malformed specs are refused with a reason and leave the table empty (opposite-answer controls)
        CHECK(!t.parse("low", err) && t.empty() && !err.empty());
        CHECK(!t.parse("low=", err) && t.empty());
        CHECK(!t.parse("=0.5", err) && t.empty());
        CHECK(!t.parse("low=abc", err) && t.empty());
        CHECK(!t.parse("low=0", err) && t.empty());     // 0 would mean "no thinking", which is effort none
        CHECK(!t.parse("low=1.5", err) && t.empty());   // more than the whole generation
        CHECK(!t.parse("low=-0.1", err) && t.empty());
        CHECK(!t.parse("low=nan", err) && t.empty());
        CHECK(!t.parse("", err) && t.empty());
        CHECK(t.parse("low=100%", err) && t.budget("low", 50) == 50);   // exactly the whole generation is allowed
    }

    // ---- the effective limit -------------------------------------------------------------------------------------------
    CHECK(server_effort_table::effective_limit(400, -1)  == 400);   // request only
    CHECK(server_effort_table::effective_limit(-1, 256)  == 256);   // server only
    CHECK(server_effort_table::effective_limit(400, 256) == 256);   // the server caps the request
    CHECK(server_effort_table::effective_limit(100, 256) == 100);
    CHECK(server_effort_table::effective_limit(-1, -1)   == -1);
    CHECK(server_effort_table::effective_limit(0, 0)     == -1);

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("test_server_effort: all checks passed\n");
    return 0;
}
