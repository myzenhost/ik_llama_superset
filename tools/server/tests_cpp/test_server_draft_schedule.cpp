// Standalone test for server-draft-schedule.h. Build: g++ -std=c++17 -I.. test_server_draft_schedule.cpp -o t && ./t
#include "server-draft-schedule.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static bool bad(const char * spec) {
    server_draft_schedule s;
    std::string err;
    const bool ok = s.parse(spec, err);
    CHECK(!ok == !err.empty());
    CHECK(s.empty() || ok); // a refused spec leaves nothing behind
    return !ok;
}

int main() {
    // ---- the row's table: '1:4,2:0' -------------------------------------------------------------------------------------
    {
        server_draft_schedule s;
        std::string err;
        CHECK(s.parse("1:4,2:0", err));
        CHECK(s.lookup(1) == 4);
        CHECK(s.lookup(2) == 0);
        CHECK(s.lookup(7) == 0);   // past the table: the last K carries
        CHECK(s.lookup(0) == -1);  // nobody generating: no opinion
        CHECK(s.clamp(8, 1) == 4); // lowers
        CHECK(s.clamp(3, 1) == 3); // never raises
        CHECK(s.clamp(8, 2) == 0); // no drafting
    }

    // ---- thresholds and ranges, gap carry, open tail --------------------------------------------------------------------
    {
        server_draft_schedule s;
        std::string err;
        CHECK(s.parse("1:5,3:2,9:0", err));
        CHECK(s.lookup(1) == 5 && s.lookup(2) == 5);
        CHECK(s.lookup(3) == 2 && s.lookup(8) == 2);
        CHECK(s.lookup(9) == 0 && s.lookup(100) == 0);
    }
    {
        server_draft_schedule s;
        std::string err;
        CHECK(s.parse("1-4:5, 8-16:2", err)); // spaces are fine
        CHECK(s.lookup(4) == 5);
        CHECK(s.lookup(5) == 5 && s.lookup(7) == 5); // the gap carries the previous K (a gap is neither an error nor 0)
        CHECK(s.lookup(8) == 2 && s.lookup(16) == 2);
        CHECK(s.lookup(17) == 2 && s.lookup(1000) == 2); // the tail carries the last K
    }
    {
        server_draft_schedule s;
        std::string err;
        CHECK(s.parse("1:3", err)); // a single entry covers every count
        CHECK(s.lookup(1) == 3 && s.lookup(64) == 3);
    }

    // ---- hysteresis: a rise needs H slots of margin, a drop is immediate ----------------------------------------------------
    {
        server_draft_schedule s;
        std::string err;
        CHECK(s.parse("1-2:5,3:2", err));
        s.hysteresis = 1;
        CHECK(s.lookup_sticky(1) == 5);
        CHECK(s.lookup_sticky(3) == 2);  // drop is immediate
        CHECK(s.lookup_sticky(2) == 2);  // raw 5, but 3 is still in the K=2 range: held
        CHECK(s.lookup_sticky(3) == 2);
        CHECK(s.lookup_sticky(1) == 5);  // 2 is still K=5: margin of 1 slot reached
        CHECK(s.lookup_sticky(2) == 5);  // no change at the edge going the other way (a rise was already taken)
        CHECK(s.lookup_sticky(3) == 2);  // drop again, immediate
        CHECK(s.lookup_sticky(0) == -1); // no generating slot: no opinion, state reset
        CHECK(s.lookup_sticky(2) == 5);  // fresh state takes the raw value
    }
    {
        server_draft_schedule s;
        std::string err;
        CHECK(s.parse("1-2:5,3:2", err)); // hysteresis 0 (default) = plain lookup
        CHECK(s.lookup_sticky(3) == 2 && s.lookup_sticky(2) == 5);
    }

    // ---- refusals: unsorted, overlapping, first not 1, junk ---------------------------------------------------------------
    CHECK(bad("3:2,1:5"));      // unsorted (the source would sort it; this port refuses, see SOURCE-NUANCE 1)
    CHECK(bad("2:2,3:1"));      // first entry does not start at 1
    CHECK(bad("1-4:5,3-6:2"));  // overlap
    CHECK(bad("1-4:5,4:2"));    // next starts inside the previous range
    CHECK(bad("1:5,1:2"));      // repeat
    CHECK(bad("1:-1"));         // negative K
    CHECK(bad("0:3"));          // slot counts start at 1
    CHECK(bad("4-2:1"));        // start above end
    CHECK(bad("1:x"));
    CHECK(bad("1"));            // no K
    CHECK(bad(""));
    CHECK(bad("1:2,,3:1"));     // empty entry

    // ---- a refused spec after a good one leaves the schedule empty (the caller keeps the old object on failure) ---------------
    {
        server_draft_schedule s;
        std::string err;
        CHECK(s.parse("1:4", err));
        CHECK(!s.parse("junk", err));
        CHECK(s.empty() && s.lookup(1) == -1 && s.clamp(6, 1) == 6); // empty = no opinion = the budget the slot had
    }

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("All tests passed\n");
    return 0;
}
