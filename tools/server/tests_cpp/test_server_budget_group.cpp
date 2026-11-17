// Standalone test for server-budget-group.h. Build: g++ -std=c++17 -I.. test_server_budget_group.cpp -o t && ./t
// No model, no server process. Each block carries an opposite-answer control.
// RESEARCH-DONE: arxiv.org/abs/2512.24601 (RLM paper; evidence in temp/research/findings/engrev/rlm-paper/EVIDENCE.md)
#include "server-budget-group.h"
#include "server-request-log.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // ---- ingestion ----------------------------------------------------------------------------------------------
    {
        auto a = server_budget_ingest("tree-1", "", "500", "", "2000", "");
        CHECK(a.valid && a.id == "tree-1" && a.limits.max_tokens == 500 && a.limits.max_wall_ms == 2000);
        auto b = server_budget_ingest("", "body-id", "", "7", "", "");          // body fields work without headers
        CHECK(b.valid && b.id == "body-id" && b.limits.max_tokens == 7 && b.limits.max_wall_ms == 0);
        auto c = server_budget_ingest("hdr", "body", "", "", "", "");           // header beats body
        CHECK(c.valid && c.id == "hdr");
        // controls: each malformed shape drops the group entirely (never half-applied)
        CHECK(!server_budget_ingest("", "", "5", "", "", "").valid);            // no id
        CHECK(!server_budget_ingest("a b", "", "5", "", "", "").valid);         // space in the id
        CHECK(!server_budget_ingest("a\nb", "", "5", "", "", "").valid);        // newline in the id
        CHECK(!server_budget_ingest(std::string(129, 'x'), "", "5", "", "", "").valid);
        CHECK(!server_budget_ingest("g", "", "-5", "", "", "").valid);          // negative
        CHECK(!server_budget_ingest("g", "", "5x", "", "", "").valid);          // junk
        CHECK(!server_budget_ingest("g", "", "", "", "9999999999999", "").valid); // 13 digits
        // the id rule is the same one the request id uses
        CHECK(server_request_id_ok("tree-1") && server_budget_ingest("tree-1", "", "", "", "", "").valid);
        CHECK(!server_request_id_ok("a b") && !server_budget_ingest("a b", "", "", "", "", "").valid);
    }
    // ---- token pool shared by three sub-calls ------------------------------------------------------------------
    {
        server_budget_groups g;
        server_budget_limits lim; lim.max_tokens = 10;
        int64_t t = 1000000;
        g.open("tree", lim, t);
        g.open("tree", lim, t);
        int64_t total = 0;
        // three interleaved sub-calls: each asks for 100 tokens, together they get about 10
        bool done[3] = {false, false, false};
        for (int step = 0; step < 100; ++step) {
            for (int k = 0; k < 3; ++k) {
                if (done[k]) { continue; }
                ++total;
                if (g.charge("tree", 1, t) != SERVER_BUDGET_OK) { done[k] = true; }
            }
        }
        // the cap-th token is the last allowed; the others each stop on their next token (bounded overshoot of one per call)
        CHECK(total >= 10 && total <= 12);
        CHECK(g.check("tree", t) == SERVER_BUDGET_TOKENS);
        // control: an unrelated group and a group without a pool are untouched
        g.open("other", lim, t);
        CHECK(g.check("other", t) == SERVER_BUDGET_OK && g.charge("other", 3, t) == SERVER_BUDGET_OK);
        CHECK(g.charge("never-opened", 999, t) == SERVER_BUDGET_OK);
        // a later request cannot raise the cap of an existing tree
        server_budget_limits big; big.max_tokens = 1000000;
        g.open("tree", big, t);
        CHECK(g.check("tree", t) == SERVER_BUDGET_TOKENS);
        CHECK(g.snapshot("tree", t).max_tokens == 10);
    }
    // ---- wall-time pool counts from the first request, and is exact at the edge --------------------------------
    {
        server_budget_groups g;
        server_budget_limits lim; lim.max_wall_ms = 100;
        g.open("w", lim, 5000000);
        CHECK(g.check("w", 5000000 + 99999) == SERVER_BUDGET_OK);              // 99.999 ms: still inside
        CHECK(g.check("w", 5000000 + 100000) == SERVER_BUDGET_WALL);           // 100 ms: out
        CHECK(g.charge("w", 1, 5000000 + 100000) == SERVER_BUDGET_WALL);
        // control: a pool with no limits has no cap
        server_budget_groups h;
        h.open("free", server_budget_limits{}, 0);
        CHECK(h.charge("free", 1000000, 1ll << 50) == SERVER_BUDGET_OK);
        CHECK(!server_budget_limits{}.any());
        // token cap is reported first when both trip
        server_budget_limits both; both.max_tokens = 1; both.max_wall_ms = 1;
        h.open("both", both, 0);
        CHECK(h.charge("both", 1, 5000) == SERVER_BUDGET_TOKENS);
    }
    // ---- bounded table: idle pools are purged, a full table never evicts a live tree --------------------------
    {
        server_budget_groups g;
        g.set_limits(2, 1000000);
        server_budget_limits lim; lim.max_tokens = 5;
        g.open("a", lim, 0);
        g.open("b", lim, 0);
        g.open("c", lim, 10);                                                  // full: "c" stays uncapped, "a" and "b" live
        CHECK(g.size() == 2 && !g.snapshot("c", 10).known);
        CHECK(g.charge("c", 100, 10) == SERVER_BUDGET_OK);
        g.open("c", lim, 2000000);                                             // a and b idle past the ttl: purged, c fits
        CHECK(g.size() == 1 && g.snapshot("c", 2000000).known && !g.snapshot("a", 2000000).known);
    }
    // ---- request log carries the group id (and omits the field without one) -----------------------------------
    {
        server_request_record r;
        r.request_id = "r1"; r.status = "budget_group"; r.budget_group = "tree-1";
        const std::string j = r.to_jsonl();
        CHECK(j.find("\"budget_group\":\"tree-1\"") != std::string::npos && j.find("\"status\":\"budget_group\"") != std::string::npos);
        server_request_record n;
        n.request_id = "r2";
        CHECK(n.to_jsonl().find("budget_group") == std::string::npos);
    }
    // ---- stop reason strings ------------------------------------------------------------------------------------
    CHECK(std::string(server_budget_stop_str(SERVER_BUDGET_TOKENS)) == "tokens");
    CHECK(std::string(server_budget_stop_str(SERVER_BUDGET_WALL)) == "wall");
    CHECK(std::string(server_budget_stop_str(SERVER_BUDGET_OK)).empty());

    if (fails) {
        std::printf("%d FAILED\n", fails);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
