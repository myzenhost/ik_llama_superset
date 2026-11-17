// Server tools output budget: router mode and model swap.
//  - the router process runs /tools itself and loads no model; its budget comes from the running children's contexts (the child the
//    body's "model" names, else the largest), not the 16 KiB no-model fallback;
//  - the budget follows a context change between calls (a model swap / num_ctx reload / wake from sleep), it is not frozen at the first load;
//  - --tools-max-output still wins.
// Drives server_tools::setup() + handle_post() like the HTTP route. Row S69-SERVER-TOOLS-BUDGET-ROUTER-AND-HOTSWAP-2026-11-09-01.
// Old-logic build: this file compiled against a tree without SERVER_TOOLS_HAS_CTX_PROVIDER leaves the router with no provider (what
// the old router had: n_ctx_seq stays 0), so the router checks below FAIL there.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "server-tools.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static const std::function<bool()> never_stop = []() { return false; };

static json call(server_tools & t, const std::string & tool, const json & params, const std::string & model = "") {
    json body = {{"tool", tool}, {"params", params}};
    if (!model.empty()) {
        body["model"] = model;
    }
    server_http_req req { {}, {}, "/tools", "", body.dump(), {}, never_stop };
    auto res = t.handle_post(req);
    return json::parse(res->data);
}

// the running children of the pretend router: name -> per-slot context
static std::map<std::string, int64_t> g_children;

static void wire_router(server_tools & t) {
#ifdef SERVER_TOOLS_HAS_CTX_PROVIDER
    t.n_ctx_provider = [](const std::string & model) -> int64_t {
        std::vector<std::pair<std::string, int64_t>> running(g_children.begin(), g_children.end());
        return server_tools_pick_ctx(running, model);
    };
#else
    (void) t; // old router: nothing derives a context, n_ctx_seq stays 0
#endif
}

static bool whole_read_ok(server_tools & t, const std::string & path, const std::string & model, std::string * why = nullptr) {
    json r = call(t, "read_file", {{"path", path}}, model);
    if (why) {
        *why = r.dump().substr(0, 140);
    }
    return r.contains("plain_text_response") && r["plain_text_response"].get<std::string>().size() == 40000;
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "tbudget1109";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    const fs::path f40k = dir / "f40k.txt"; // 40000 bytes, 500 lines of 80
    {
        std::ofstream o(f40k, std::ios::binary);
        for (int i = 0; i < 500; i++) {
            o << std::string(79, (char) ('a' + i % 26)) << "\n";
        }
    }
    const std::string p = f40k.string();

    server_mcp mcp;
    server_tools t;
    t.setup({"read_file"}, mcp, "");

    // A. router, no child running: the 16384 fallback (a 40000-byte file is refused, the error names the budget)
    wire_router(t);
    g_children.clear();
    {
        std::string why;
        CHECK(!whole_read_ok(t, p, "", &why), "no child running must keep the fallback budget");
        CHECK(why.find("max 16384") != std::string::npos, "no child: want 'max 16384', got %s", why.c_str());
    }

    // B. router, two running children (8192 and 131072), no route known: the largest one decides (65536 holds the file)
    g_children = {{"small", 8192}, {"big", 131072}};
    {
        std::string why;
        CHECK(whole_read_ok(t, p, "", &why), "router with a 131072-ctx child refused a 40000-byte read: %s", why.c_str());
    }

    // C. the body's "model" routes the budget to that child: small = 8192 -> 4096 bytes
    {
        std::string why;
        CHECK(!whole_read_ok(t, p, "small", &why), "model=small (ctx 8192) must not hold 40000 bytes");
        CHECK(why.find("max 4096") != std::string::npos, "model=small: want 'max 4096', got %s", why.c_str());
        CHECK(whole_read_ok(t, p, "big", &why), "model=big (ctx 131072) refused: %s", why.c_str());
        // an unknown name = no route known = the largest running child
        CHECK(whole_read_ok(t, p, "nope", &why), "unknown model must fall back to the largest child: %s", why.c_str());
    }

    // D. swap: the big child is replaced by a smaller one between two calls; the next call sees the new context, no restart
    g_children = {{"small", 8192}};
    {
        std::string why;
        CHECK(!whole_read_ok(t, p, "", &why), "after the swap to ctx 8192 the budget stayed stale: %s", why.c_str());
        CHECK(why.find("max 4096") != std::string::npos, "after the swap: want 'max 4096', got %s", why.c_str());
    }
    g_children = {{"small", 8192}, {"big", 262144}};
    {
        std::string why;
        CHECK(whole_read_ok(t, p, "", &why), "after a swap back to a roomy child the read is refused: %s", why.c_str());
    }

    // E. --tools-max-output wins over the router-derived context
    t.output_budget_flag.store(2048);
    {
        std::string why;
        CHECK(!whole_read_ok(t, p, "", &why), "flag 2048 must beat child ctx 262144");
        CHECK(why.find("max 2048") != std::string::npos, "flag: want 'max 2048', got %s", why.c_str());
    }
    t.output_budget_flag.store(0);

    // F. single-model server, model reloaded at another context (wake from sleep): the callback in server.cpp stores the new
    //    n_ctx_seq; the very next call must use it
    {
        server_tools s;
        s.setup({"read_file"}, mcp, "");
        std::string why;
        s.n_ctx_seq.store(131072); // first load
        CHECK(whole_read_ok(s, p, "", &why), "first load ctx 131072: %s", why.c_str());
        s.n_ctx_seq.store(8192);   // reload at a smaller context
        CHECK(!whole_read_ok(s, p, "", &why) && why.find("max 4096") != std::string::npos, "reload ctx 8192 not honoured: %s", why.c_str());
    }

#ifdef SERVER_TOOLS_HAS_CTX_PROVIDER
    // G. the pure picker
    {
        std::vector<std::pair<std::string, int64_t>> r = {{"a", 4096}, {"b", 32768}, {"c", 0}};
        CHECK(server_tools_pick_ctx(r, "a") == 4096, "route a");
        CHECK(server_tools_pick_ctx(r, "")  == 32768, "no route = largest");
        CHECK(server_tools_pick_ctx(r, "c") == 32768, "a child whose ctx is unreadable (0) is not a route");
        CHECK(server_tools_pick_ctx({}, "a") == 0, "none running = 0");
    }
#endif

    fs::remove_all(dir, ec);
    if (fails) {
        fprintf(stderr, "%d check(s) FAILED\n", fails);
        return 1;
    }
    printf("ok\n");
    return 0;
}
