// Standalone test for server-load-fail.h. Build: g++ -std=c++17 -I.. test_server_load_fail.cpp -o t && ./t
#include "server-load-fail.h"

#include <cstdio>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // ---- the ring keeps the LAST n lines, oldest first ------------------------------------------------------------------
    {
        server_output_ring r(3);
        CHECK(r.text().empty());
        r.push("one\n");
        r.push("two\r\n");
        CHECK(r.text() == "one\ntwo");
        r.push("three");
        r.push("four");
        CHECK(r.text() == "two\nthree\nfour"); // "one" fell off
    }

    // ---- blank lines are not stored, long lines are clipped at the head ------------------------------------------------
    {
        server_output_ring r(4, 5);
        r.push("\n");
        r.push("   \t\n");
        CHECK(r.text().empty());
        r.push("abcdefghij");
        CHECK(r.text() == "abcde");
    }

    // ---- default sizes -------------------------------------------------------------------------------------------------
    {
        server_output_ring r;
        for (int i = 0; i < 20; ++i) {
            r.push("l" + std::to_string(i));
        }
        const std::string t = r.text();
        CHECK(t.rfind("l12\n", 0) == 0); // 20 pushed, the last 8 kept: l12..l19
        CHECK(t.size() >= 3 && t.compare(t.size() - 3, 3, "l19") == 0);
    }

    // ---- the row's case: a child that exits 3 with no error-prefix line ------------------------------------------------
    {
        server_output_ring r;
        r.push("load_tensors: loading model tensors");
        r.push("llama_model_loader: done");
        const std::string d = server_load_failure_detail("", r.text(), 3);
        CHECK(d.find("exit code 3") == 0);
        CHECK(d.find("last output: load_tensors: loading model tensors|llama_model_loader: done") != std::string::npos);
        CHECK(d.find('\n') == std::string::npos); // a single line
    }

    // ---- a child that dies without writing anything still reports the exit code ----------------------------------------
    CHECK(server_load_failure_detail("", "", -1073741819) == "exit code -1073741819"); // 0xC0000005 as int
    CHECK(server_load_failure_detail("", "", 1) == "exit code 1");

    // ---- matched error lines win; they already say why ------------------------------------------------------------------
    CHECK(server_load_failure_detail("error: bad tensor", "noise", 3) == "error: bad tensor");

    // ---- a clean exit with nothing matched reports nothing -------------------------------------------------------------
    CHECK(server_load_failure_detail("", "bye", 0).empty());

    if (fails == 0) {
        std::printf("test_server_load_fail: PASS\n");
    }
    return fails == 0 ? 0 : 1;
}
