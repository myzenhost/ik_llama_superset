// Standalone test for server-auth.h. Build: g++ -std=c++17 -I.. test_server_auth.cpp -o t && ./t [path/to/server.cpp]
// No server, no model. Each block carries an opposite-answer control.
#include "server-auth.h"

#include <cstdio>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static std::string upper(std::string s) { for (auto & c : s) c = (char) std::toupper((unsigned char) c); return s; }

int main(int argc, char ** argv) {
    // ---- constant-time equality is still equality -----------------------------------------------------------------
    CHECK(server_auth_ct_equal("abc", "abc"));
    CHECK(!server_auth_ct_equal("abc", "abd"));
    CHECK(!server_auth_ct_equal("abc", "abcd"));      // a prefix is not equal
    CHECK(!server_auth_ct_equal("abcd", "abc"));
    CHECK(!server_auth_ct_equal("", "a"));
    CHECK(server_auth_ct_equal("", ""));
    CHECK(!server_auth_ct_equal(std::string("a\0b", 3), std::string("a\0c", 3)));   // embedded NUL does not end the compare
    CHECK(server_auth_ct_equal(std::string("a\0b", 3), std::string("a\0b", 3)));

    // ---- tiers ---------------------------------------------------------------------------------------------------------
    {
        server_auth a;
        CHECK(!a.enabled() && a.verdict("anything", true) == 0 && a.verdict("", false) == 0);   // not configured: open, as today

        a.api = {"user1", "user2"};
        CHECK(a.enabled() && !a.tiers_active());
        // only regular keys: every valid key opens every route (nothing changes without an admin key)
        CHECK(a.verdict("user1", false) == 0);
        CHECK(a.verdict("user1", true)  == 0);
        CHECK(a.verdict("nope", false)  == 401);
        CHECK(a.verdict("", false)      == 401);

        a.admin = {"root"};
        CHECK(a.tiers_active());
        CHECK(a.verdict("user1", false) == 0);          // regular key: inference routes
        CHECK(a.verdict("user1", true)  == 403);        // regular key on an admin route: authenticated, not permitted
        CHECK(a.verdict("root", true)   == 0);          // admin key: management routes
        CHECK(a.verdict("root", false)  == 0);          // and everywhere a regular key works (superset)
        CHECK(a.verdict("nope", true)   == 401);        // unknown key is 401, never 403
        CHECK(a.classify("root") == SERVER_AUTH_ADMIN && a.classify("user2") == SERVER_AUTH_API && a.classify("x") == SERVER_AUTH_NONE);

        // a key listed as both is admin
        server_auth b;
        b.api = {"k"}; b.admin = {"k"};
        CHECK(b.classify("k") == SERVER_AUTH_ADMIN);

        // several headers: the best level among the candidates counts (X-Admin-Key beside a regular Authorization)
        CHECK(a.verdict_multi({"user1", "root"}, true)  == 0);
        CHECK(a.verdict_multi({"root", "user1"}, true)  == 0);
        CHECK(a.verdict_multi({"user1", ""}, true)      == 403);
        CHECK(a.verdict_multi({"", ""}, false)          == 401);
        CHECK(a.verdict_multi({"nope", "user2"}, false) == 0);          // one valid candidate is enough
        CHECK(a.verdict_multi({"nope", "nope2"}, false) == 401);

        // only admin keys configured: regular routes need an admin key too
        server_auth c;
        c.admin = {"root"};
        CHECK(c.verdict("root", false) == 0 && c.verdict("other", false) == 401);
    }

    // ---- replacing keys: add/revoke without restart, but never into an empty set -------------------------------------
    {
        server_auth a;
        a.api = {"old"};
        CHECK(a.verdict("old", false) == 0);
        CHECK(a.replace({"new"}, {}));
        CHECK(a.verdict("new", false) == 0 && a.verdict("old", false) == 401);          // pulled key stops working
        CHECK(!a.replace({}, {}));                                                      // an empty re-read is refused ...
        CHECK(a.verdict("new", false) == 0);                                            // ... and the keys stay
        CHECK(a.replace({}, {"root"}) && a.verdict("root", false) == 0);                // admin-only is not empty
        server_auth fresh;
        CHECK(fresh.replace({}, {}));                                                   // from nothing to nothing is no change
        CHECK(!fresh.enabled());
    }

    // ---- key files -----------------------------------------------------------------------------------------------------
    {
        const char * path = "_test_server_auth_keys.tmp";
        {
            std::ofstream f(path, std::ios::binary);
            f << "# comment\r\n\r\n  key-one  \r\nkey-two\n\t\n#key-three\nkey #four\n";
        }
        std::vector<std::string> keys;
        CHECK(server_auth_read_key_file(path, keys));
        CHECK(keys.size() == 3);
        CHECK(keys.size() == 3 && keys[0] == "key-one" && keys[1] == "key-two" && keys[2] == "key #four");   // # only comments a LINE
        std::vector<std::string> none;
        CHECK(!server_auth_read_key_file("_no_such_key_file.tmp", none) && none.empty());     // unreadable is reported, not empty

        // the watcher sees a change of content (size) and a removal, and is quiet otherwise
        server_auth_file_watch w;
        w.init({path});
        CHECK(!w.changed());
        { std::ofstream f(path, std::ios::binary | std::ios::app); f << "key-five\n"; }
        CHECK(w.changed());
        CHECK(!w.changed());                                                                  // consumed
        std::remove(path);
        CHECK(w.changed());                                                                   // removal is a change (the caller then refuses an empty set)
    }

    // ---- admin route matching ------------------------------------------------------------------------------------------
    CHECK(server_auth_is_admin_route("POST", "/lora-adapters"));
    CHECK(!server_auth_is_admin_route("GET", "/lora-adapters"));                // listing is read-only
    CHECK(server_auth_is_admin_route("POST", "/slots/3?action=save"));          // :id_slot wildcard, query ignored
    CHECK(!server_auth_is_admin_route("GET", "/slots"));
    CHECK(!server_auth_is_admin_route("GET", "/slots/list"));
    CHECK(!server_auth_is_admin_route("POST", "/v1/chat/completions"));
    CHECK(!server_auth_is_admin_route("POST", "/completion"));
    CHECK(server_auth_is_admin_route("DELETE", "/models"));
    CHECK(!server_auth_is_admin_route("GET", "/models"));
    CHECK(server_auth_is_admin_route("POST", "/api/v1/props", "/api/v1"));      // api prefix stripped
    CHECK(!server_auth_is_admin_route("POST", "/props/extra"));                 // no prefix match on longer paths

    // ---- the route table must equal the routes server.cpp wraps with mgmt(...) ---------------------------------------
    {
        std::string src_path = argc > 1 ? argv[1] : "../server.cpp";
        std::ifstream f(src_path, std::ios::binary);
        if (!f) {
            std::printf("FAIL cannot open %s\n", src_path.c_str());
            ++fails;
        } else {
            std::stringstream ss; ss << f.rdbuf();
            const std::string text = ss.str();
            std::set<std::string> in_source, in_table;
            std::regex re(R"re(ctx_http\.(get|post|del)\s*\(\s*"([^"]+)"\s*,\s*ex_wrapper\(mgmt\()re");
            for (std::sregex_iterator it(text.begin(), text.end(), re), end; it != end; ++it) {
                std::string m = (*it)[1].str() == "del" ? "DELETE" : upper((*it)[1].str());
                in_source.insert(m + " " + (*it)[2].str());
            }
            for (const auto & r : server_auth_admin_routes()) {
                in_table.insert(std::string(r.method) + " " + r.pattern);
            }
            CHECK(!in_source.empty());                                          // positive control: the scan found routes
            for (const auto & s : in_source) {
                if (!in_table.count(s)) { std::printf("  in server.cpp mgmt() but not in the table: %s\n", s.c_str()); ++fails; }
            }
            for (const auto & s : in_table) {
                if (!in_source.count(s)) { std::printf("  in the table but not wrapped in server.cpp: %s\n", s.c_str()); ++fails; }
            }
            std::printf("route table: %zu in server.cpp, %zu in table\n", in_source.size(), in_table.size());
        }
    }

    // ---- --models-register-allow: which repos a runtime registration (router POST /models) may download -----------------
    // SOURCE-READ: upstream/torchserve/frontend/archive/src/main/java/org/pytorch/serve/archive/utils/ArchiveUtils.java:75-90
    // and .../archive/model/ModelArchive.java:46-60 (pytorch/serve@62c4d6a1, Apache-2.0).
    // SOURCE-NUANCE: each pattern is a case-insensitive FULL match (Matcher.matches, not find), and ".." in the source is refused
    // before any pattern is tried, so a matching pattern cannot let a traversal through.
    {
        const std::vector<std::string> none;
        CHECK(server_models_register_refusal(none, "acme/Qwen3-8B-GGUF").empty());                 // no list: every repo, as before
        CHECK(server_models_register_refusal(none, "acme/Qwen3-8B-GGUF:Q4_K_M").empty());
        CHECK(!server_models_register_refusal(none, "evil/../../etc").empty());                    // ".." refused even with no list
        CHECK(!server_models_register_refusal(none, "a/b..c").empty());

        const std::vector<std::string> allow = {"acme/.*", "beta/Qwen3-.*-GGUF(:.*)?"};
        CHECK(server_models_register_refusal(allow, "acme/Qwen3-8B-GGUF").empty());
        CHECK(server_models_register_refusal(allow, "ACME/qwen3-8b-gguf").empty());                // case-insensitive, like TorchServe
        CHECK(server_models_register_refusal(allow, "beta/Qwen3-30B-A3B-GGUF:Q8_0").empty());
        CHECK(!server_models_register_refusal(allow, "beta/gemma-3-GGUF").empty());                // opposite-answer control
        CHECK(!server_models_register_refusal(allow, "xacme/Qwen3").empty());                      // FULL match, not a substring
        CHECK(!server_models_register_refusal(allow, "mirror/acme/Qwen3").empty());
        CHECK(!server_models_register_refusal(allow, "acme/../beta/x").empty());                   // ".." wins over a matching pattern
        CHECK(server_models_register_refusal(allow, "evil/x").find("evil/x") != std::string::npos); // the reason names the repo

        std::string err;
        CHECK(server_models_register_pattern_ok("acme/.*", err) && err.empty());
        CHECK(!server_models_register_pattern_ok("acme/[", err) && !err.empty());                  // a bad regex is caught at start-up
        CHECK(!server_models_register_pattern_ok("", err));                                         // an empty pattern is refused
        const std::vector<std::string> bad = {"acme/["};
        CHECK(!server_models_register_refusal(bad, "acme/x").empty());                              // backstop: refuses, never throws
    }

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("test_server_auth: all checks passed\n");
    return 0;
}
