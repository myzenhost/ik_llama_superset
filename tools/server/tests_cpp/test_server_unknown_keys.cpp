// Unit test for server-unknown-keys.h (one warning per unknown request key). Model-free; links llama-common for the JSON class.
// RESEARCH-DONE: behaviour spec from TabbyAPI common/sampling.py:428-449 (see server-unknown-keys.h SOURCE-READ).
#include "server-unknown-keys.h"

#include <cstdio>
#include <set>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    const std::set<std::string> known = { "temperature", "top_p", "rep_pen", "stream" };
    const auto is_known = [&](const std::string & k) { return known.count(k) > 0; };

    // a typo is reported, in body order; known keys are not
    {
        const auto u = server_unknown_request_keys(common_json::parse(R"({"top_P":0.1,"temperature":0,"zzz":1,"rep_pen":1.8})"), is_known);
        CHECK(u.size() == 2);
        CHECK(u.size() == 2 && u[0] == "top_P" && u[1] == "zzz");
    }
    // control: the same body with the typo fixed has nothing to report (the check can say "none")
    {
        const auto u = server_unknown_request_keys(common_json::parse(R"({"top_p":0.1,"temperature":0})"), is_known);
        CHECK(u.empty());
    }
    // null counts as absent; a non-object body has no keys
    {
        CHECK(server_unknown_request_keys(common_json::parse(R"({"nope":null})"), is_known).empty());
        CHECK(server_unknown_request_keys(common_json::parse("[1,2]"), is_known).empty());
        CHECK(server_unknown_request_keys(common_json::parse("\"s\""), is_known).empty());
    }
    // registry: first sighting is new, a repeat is not, a different key is
    {
        server_unknown_key_registry reg(256);
        bool cap = false;
        auto a = reg.take_new({ "top_P" }, cap);
        CHECK(a.size() == 1 && a[0] == "top_P" && !cap);
        auto b = reg.take_new({ "top_P" }, cap);
        CHECK(b.empty());
        auto c = reg.take_new({ "top_P", "other", "other" }, cap);
        CHECK(c.size() == 1 && c[0] == "other");
    }
    // registry cap: memory is bounded, the cap is announced once
    {
        server_unknown_key_registry reg(3);
        bool cap = false;
        auto a = reg.take_new({ "a", "b", "c", "d" }, cap);
        CHECK(a.size() == 3);
        CHECK(cap);
        auto b = reg.take_new({ "e" }, cap);
        CHECK(b.empty());
        CHECK(!cap);
        auto c = reg.take_new({ "a" }, cap);  // already-seen keys stay quiet
        CHECK(c.empty());
    }
    // log sanitising: control characters cannot forge a line, length is capped
    {
        CHECK(server_unknown_key_for_log("a\nb\r") == "a?b?");
        const std::string longk(100, 'x');
        const std::string out = server_unknown_key_for_log(longk);
        CHECK(out.size() == 67 && out.substr(64) == "...");
        CHECK(server_unknown_key_for_log("top_P") == "top_P");
    }

    if (fails == 0) {
        std::printf("test-server-unknown-keys: all checks passed\n");
        return 0;
    }
    std::printf("test-server-unknown-keys: %d failure(s)\n", fails);
    return 1;
}
