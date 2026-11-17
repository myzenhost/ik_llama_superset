// Standalone test for server-answer-reserve.h. Build: g++ -std=c++17 -I.. test_server_answer_reserve.cpp -o t && ./t
// Row S69-REASONING-ANSWER-RESERVE-OPT-IN-2026-10-17-01.
#include "server-answer-reserve.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // ---- parsing: off, tokens, percent -------------------------------------------------------------------------------
    {
        server_answer_reserve r;
        std::string err;
        CHECK(!r.active());                                 // default: off
        CHECK(r.parse("", err) && !r.active());             // empty = off
        CHECK(r.parse("-1", err) && !r.active());           // -1 = off (the flag default)
        CHECK(r.parse("off", err) && !r.active());
        CHECK(r.parse("64", err) && r.active() && r.tokens == 64 && r.fraction < 0);
        CHECK(r.parse(" 64 ", err) && r.tokens == 64);      // surrounding spaces tolerated
        CHECK(r.parse("0", err) && r.active() && r.tokens == 0);   // reserve nothing: budget = whole limit
        CHECK(r.parse("25%", err) && r.active() && r.tokens < 0 && r.fraction == 0.25);
        CHECK(r.parse("12.5%", err) && r.fraction == 0.125);
        CHECK(r.parse("100%", err) && r.fraction == 1.0);   // the whole limit for the answer: thinking forced shut
        CHECK(r.parse("0%", err) && r.fraction == 0.0);

        // malformed specs are refused with a reason and leave the reserve off (opposite-answer controls)
        CHECK(!r.parse("abc", err) && !r.active() && !err.empty());
        CHECK(!r.parse("25%%", err) && !r.active());
        CHECK(!r.parse("%", err) && !r.active());
        CHECK(!r.parse("101%", err) && !r.active());
        CHECK(!r.parse("-5%", err) && !r.active());
        CHECK(!r.parse("-2", err) && !r.active());
        CHECK(!r.parse("1.5", err) && !r.active());         // a token count is a whole number
        CHECK(!r.parse("nan%", err) && !r.active());
        CHECK(!r.parse("64tok", err) && !r.active());
    }

    // ---- the derived budget ------------------------------------------------------------------------------------------
    {
        server_answer_reserve r;
        std::string err;
        CHECK(r.budget(256) == -1);                         // off: nothing derived

        CHECK(r.parse("25%", err));
        CHECK(r.budget(256) == 192);                        // 256 - 64
        CHECK(r.budget(100) == 75);
        CHECK(r.budget(10)  == 7);                          // reserve = ceil(2.5) = 3: the answer gets at least P%
        CHECK(r.budget(1)   == 0);                          // ceil(0.25) = 1 reserved, floored at 0
        CHECK(r.budget(-1)  == -1);                         // no finite limit: the reserve does nothing
        CHECK(r.budget(0)   == -1);

        CHECK(r.parse("100%", err) && r.budget(256) == 0);
        CHECK(r.parse("0%",   err) && r.budget(256) == 256);

        CHECK(r.parse("64", err));
        CHECK(r.budget(256) == 192);
        CHECK(r.budget(64)  == 0);                          // reserve == limit: thinking forced shut
        CHECK(r.budget(10)  == 0);                          // reserve > limit: floored at 0, never negative
        CHECK(r.budget(-1)  == -1);
    }

    // ---- precedence: request budget > effort budget > reserve-derived > server default ------------------------------
    {
        using R = server_answer_reserve;
        CHECK(R::resolve(7,  50, 192, 999) == 7);           // explicit request budget wins over everything
        CHECK(R::resolve(0,  50, 192, 999) == 0);           // an explicit 0 (thinking off) is a budget too
        CHECK(R::resolve(-1, 50, 192, 999) == 50);          // effort budget next
        CHECK(R::resolve(-1, -1, 192, 999) == 192);         // then the reserve
        CHECK(R::resolve(-1, -1, 0,   999) == 0);           // a reserve-derived 0 still beats the server default
        CHECK(R::resolve(-1, -1, -1,  999) == 999);         // then the server default
        CHECK(R::resolve(-1, -1, -1,  -1)  == -1);          // nothing at all: unrestricted, as before
        CHECK(R::resolve(-5, 50, 192, 999) == -5);          // any request value but -1 passes through unchanged (old rule)
    }

    // ---- request override of the server reserve ----------------------------------------------------------------------
    {
        server_answer_reserve server;
        std::string err;
        CHECK(server.parse("25%", err));

        // request absent: server reserve stays
        server_answer_reserve eff = server;
        CHECK(eff.budget(256) == 192);

        // request "50%" replaces it
        CHECK(eff.parse("50%", err) && eff.budget(256) == 128);
        // request 32 (tokens) replaces it
        CHECK(eff.parse("32", err) && eff.budget(256) == 224);
        // request -1 turns it off for this request even though the server has one
        CHECK(eff.parse("-1", err) && eff.budget(256) == -1);
    }

    if (fails == 0) {
        std::printf("test_server_answer_reserve: all checks passed\n");
        return 0;
    }
    std::printf("test_server_answer_reserve: %d check(s) FAILED\n", fails);
    return 1;
}
