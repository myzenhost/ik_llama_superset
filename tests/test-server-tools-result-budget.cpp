// Built-in server tools file_glob_search and grep_search: how many results go back to the model is a byte budget taken from
// the serving context (or --tools-max-output), not a fixed 100. The "limit reached" text stays when the budget cuts the list.
// Drives the real path: server_tools::setup() + handle_post(). Row S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01 (lint row D11).
// GATE-CLAIM-OK: unit test, no resource claims.
#include "server-tools.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static const std::function<bool()> never_stop = []() { return false; };

static json call(server_tools & t, const std::string & tool, const json & params) {
    server_http_req req { {}, {}, "/tools", "", json{{"tool", tool}, {"params", params}}.dump(), {}, never_stop };
    auto res = t.handle_post(req);
    return json::parse(res->data);
}

static size_t total_of(const std::string & text) {
    const size_t p = text.find("Total matches: ");
    return p == std::string::npos ? (size_t) -1 : (size_t) std::stoul(text.substr(p + 15));
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "nolim3_tools_result";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / "many");
    fs::create_directories(dir / "grep");
    for (int i = 0; i < 400; i++) {
        char nm[32];
        snprintf(nm, sizeof(nm), "f%03d.txt", i);                 // 8 chars + newline = 9 bytes per listing line
        std::ofstream(dir / "many" / nm) << "x";
    }
    {
        std::ofstream o(dir / "grep" / "g.txt");
        for (int i = 0; i < 300; i++) {
            o << "match " << (i % 10) << (i / 10 % 10) << (i / 100) << "\n"; // "g.txt:match NNN\n" = 16 bytes per hit
        }
    }

    server_mcp mcp;
    server_tools t;
    t.setup({"file_glob_search", "grep_search"}, mcp, "");

    const std::string LIM = "limit reached";

    // glob, ctx 131072: budget 64 KiB holds all 400 listing lines (3600 bytes)
    t.n_ctx_seq.store(131072);
    {
        json r = call(t, "file_glob_search", {{"path", (dir / "many").string()}});
        const std::string s = r.value("plain_text_response", std::string());
        const size_t n = r.contains("entries") ? r["entries"].size() : 0;
        CHECK(n == 400, "glob at ctx 131072 returned %zu entries, want 400", n);
        CHECK(s.find(LIM) == std::string::npos, "glob at ctx 131072 says limit reached");
        CHECK(total_of(s) == 400, "Total matches %zu", total_of(s));
    }
    // glob, ctx 1024: budget 512 bytes = 56 lines of 9 bytes; the cut is announced
    t.n_ctx_seq.store(1024);
    {
        json r = call(t, "file_glob_search", {{"path", (dir / "many").string()}});
        const std::string s = r.value("plain_text_response", std::string());
        const size_t n = r.contains("entries") ? r["entries"].size() : 0;
        CHECK(n >= 1 && n <= 512 / 9, "glob at ctx 1024 returned %zu entries, want 1..%d", n, 512 / 9);
        CHECK(s.find(LIM) != std::string::npos, "glob at ctx 1024 lost the 'limit reached' text");
        CHECK(total_of(s) == 400, "Total matches %zu, want 400 (the count is of all matches)", total_of(s));
        // a client limit below the budget still applies
        json r2 = call(t, "file_glob_search", {{"path", (dir / "many").string()}, {"limit", 5}});
        CHECK(r2.contains("entries") && r2["entries"].size() == 5, "client limit 5 gave %zu", r2.contains("entries") ? r2["entries"].size() : 0);
    }
    // glob, flag 2000 over a big ctx
    t.n_ctx_seq.store(131072);
    t.output_budget_flag.store(2000);
    {
        json r = call(t, "file_glob_search", {{"path", (dir / "many").string()}});
        const size_t n = r.contains("entries") ? r["entries"].size() : 0;
        CHECK(n >= 1 && n <= 2000 / 9, "glob with flag 2000 returned %zu entries, want 1..%d", n, 2000 / 9);
    }
    t.output_budget_flag.store(0);

    // grep, ctx 131072: all 300 hits (4800 bytes) fit
    t.n_ctx_seq.store(131072);
    {
        json r = call(t, "grep_search", {{"path", (dir / "grep").string()}, {"pattern", "match"}});
        const std::string s = r.value("plain_text_response", std::string());
        CHECK(total_of(s) == 300, "grep at ctx 131072 reported %zu matches, want 300", total_of(s));
        CHECK(s.find(LIM) == std::string::npos, "grep at ctx 131072 says limit reached");
    }
    // grep, ctx 1024: budget 512 bytes = 32 hits; the cut is announced
    t.n_ctx_seq.store(1024);
    {
        json r = call(t, "grep_search", {{"path", (dir / "grep").string()}, {"pattern", "match"}});
        const std::string s = r.value("plain_text_response", std::string());
        const size_t n = total_of(s);
        CHECK(n >= 1 && n <= 512 / 16, "grep at ctx 1024 reported %zu matches, want 1..%d", n, 512 / 16);
        CHECK(s.find(LIM) != std::string::npos, "grep at ctx 1024 lost the 'limit reached' text");
    }

    fs::remove_all(dir, ec);
    if (fails) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return 1;
    }
    printf("ok\n");
    return 0;
}
