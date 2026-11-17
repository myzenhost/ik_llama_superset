// Built-in server tools (read_file, exec_shell_command, get_info): the output budget handed back to the model comes from
// the serving context (n_ctx_seq, set once the model is loaded) or from --tools-max-output, not from a fixed 16 KiB / 4096.
// Drives the real path: server_tools::setup() + handle_post() with a JSON tool call, as the HTTP route does.
// Row S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01 (lint rows D10, D12).
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

static std::string make_file(const fs::path & dir, const char * name, size_t n_lines, size_t line_len) {
    const fs::path p = dir / name;
    std::ofstream o(p, std::ios::binary);
    for (size_t i = 0; i < n_lines; i++) {
        std::string l(line_len - 1, (char) ('a' + i % 26));
        o << l << "\n";
    }
    return p.string();
}

static const std::string MARK = "[output truncated]";

int main() {
    // A. the budget function: a fraction of n_ctx_seq in bytes, the flag wins, no context = fallback
    CHECK(server_tools_output_budget(131072, 0) == 65536, "ctx 131072 -> %zu, want 65536", server_tools_output_budget(131072, 0));
    CHECK(server_tools_output_budget(8192,   0) == 4096,  "ctx 8192 -> %zu, want 4096",   server_tools_output_budget(8192, 0));
    CHECK(server_tools_output_budget(131072, 1000) == 1000, "flag 1000 over ctx -> %zu", server_tools_output_budget(131072, 1000));
    CHECK(server_tools_output_budget(0, 0) == 16384, "no ctx -> %zu, want the 16384 fallback", server_tools_output_budget(0, 0));
    CHECK(server_tools_output_budget(1, 0) >= 1, "tiny ctx must not give a zero budget");

    const fs::path dir = fs::temp_directory_path() / "nolim3_tools_budget";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    const std::string f40k = make_file(dir, "f40k.txt", 500, 80);   // 40000 bytes, 500 lines
    const std::string f30k = make_file(dir, "f30k.txt", 300, 100);  // 30000 bytes

    server_mcp mcp;
    server_tools t;
    t.setup({"read_file", "exec_shell_command", "get_info"}, mcp, "");

    // B. read_file, whole file, ctx 131072: budget 64 KiB holds the 40000-byte file
    t.n_ctx_seq.store(131072);
    {
        json r = call(t, "read_file", {{"path", f40k}});
        CHECK(r.contains("plain_text_response"), "40000-byte file at ctx 131072 was refused: %s", r.dump().substr(0, 160).c_str());
        if (r.contains("plain_text_response")) {
            const std::string s = r["plain_text_response"].get<std::string>();
            CHECK(s.size() == 40000, "read %zu bytes, want 40000", s.size());
            CHECK(s.find(MARK) == std::string::npos, "whole read carries the truncation marker");
        }
    }
    // B2. ctx 8192: budget 4096, whole-file read refused with the size, a range read truncated with the marker
    t.n_ctx_seq.store(8192);
    {
        json r = call(t, "read_file", {{"path", f40k}});
        CHECK(r.contains("error") && r["error"].get<std::string>().find("file too large") != std::string::npos, "whole read at ctx 8192: %s", r.dump().substr(0, 160).c_str());
        CHECK(r.contains("error") && r["error"].get<std::string>().find("max 4096") != std::string::npos, "error must name the derived budget: %s", r.dump().substr(0, 160).c_str());
        json r2 = call(t, "read_file", {{"path", f40k}, {"end_line", 500}});
        const std::string s = r2.value("plain_text_response", std::string());
        CHECK(s.size() >= MARK.size() && s.substr(s.size() - MARK.size()) == MARK, "range read keeps the marker");
        CHECK(s.size() <= 4096 + MARK.size(), "range read %zu bytes, budget 4096", s.size());
    }
    // B3. --tools-max-output wins over the context
    t.n_ctx_seq.store(131072);
    t.output_budget_flag.store(2048);
    {
        json r = call(t, "read_file", {{"path", f40k}, {"end_line", 500}});
        const std::string s = r.value("plain_text_response", std::string());
        CHECK(s.size() <= 2048 + MARK.size() && s.find(MARK) != std::string::npos, "flag 2048: %zu bytes, marker %d", s.size(), (int) (s.find(MARK) != std::string::npos));
    }
    t.output_budget_flag.store(0);

    // C. exec_shell_command: output cap = min(client max_output_size, budget)
#if defined(_WIN32)
    const std::string cmd = "type " + f30k; // no quotes: cmd /c with a quoted path was mangled by the argument quoting (p1 showed an 83-byte error)
#else
    const std::string cmd = "cat '" + f30k + "'";
#endif
    t.n_ctx_seq.store(131072);
    {
        json r = call(t, "exec_shell_command", {{"command", cmd}, {"max_output_size", 1000000}});
        const std::string s = r.value("plain_text_response", std::string());
        CHECK(s.size() >= 30000, "exec at ctx 131072 returned %zu bytes, want all 30000", s.size());
        CHECK(s.find(MARK) == std::string::npos, "exec at ctx 131072 was truncated");
    }
    t.n_ctx_seq.store(8192); // budget 4096
    {
        json r = call(t, "exec_shell_command", {{"command", cmd}, {"max_output_size", 1000000}});
        const std::string s = r.value("plain_text_response", std::string());
        CHECK(s.find(MARK) != std::string::npos, "exec at ctx 8192 lost the truncation marker");
        CHECK(s.size() < 4096 + 64, "exec at ctx 8192 returned %zu bytes, budget 4096", s.size());
        // the client's own smaller max_output_size still clamps down
        json r2 = call(t, "exec_shell_command", {{"command", cmd}, {"max_output_size", 500}});
        const std::string s2 = r2.value("plain_text_response", std::string());
        CHECK(s2.size() < 500 + 64 && s2.find(MARK) != std::string::npos, "client clamp 500: %zu bytes", s2.size());
    }

    // D. get_info: the runtime-info probe reads under the same budget (ctx 16 -> 8 bytes)
    t.n_ctx_seq.store(16);
    {
        json r = call(t, "get_info", json::object());
        const std::string os = r.value("os", std::string());
        CHECK(os.find(MARK) != std::string::npos, "get_info at ctx 16 not bounded by the budget: os='%s'", os.c_str());
    }
    t.n_ctx_seq.store(131072);
    {
        json r = call(t, "get_info", json::object());
        const std::string os = r.value("os", std::string());
        CHECK(!os.empty() && os != "unknown" && os.find(MARK) == std::string::npos, "get_info at ctx 131072: os='%s'", os.c_str());
    }

    fs::remove_all(dir, ec);
    if (fails) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return 1;
    }
    printf("ok\n");
    return 0;
}
