// Standalone test for server-hostguard.h. Build: g++ -std=c++17 -I.. test_server_hostguard.cpp -o t && ./t
// Every refusal is paired with the opposite request that must pass (so a guard that refuses everything fails here).
#include "server-hostguard.h"
#include <cstdio>
#include <string>

using hostguard::check;
using hostguard::config;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
#define ALLOWED(cfg, m, h, o) CHECK(check(cfg, m, h, o).empty())
#define REFUSED(cfg, m, h, o) CHECK(!check(cfg, m, h, o).empty())

int main() {
    config c; // loopback bind, no extra hosts, no cors opt-in

    // ---- Host header (DNS rebinding) ----
    ALLOWED(c, "POST", "127.0.0.1:8080", "");
    ALLOWED(c, "POST", "localhost:8080", "");
    ALLOWED(c, "POST", "LOCALHOST", "");
    ALLOWED(c, "POST", "[::1]:8080", "");
    ALLOWED(c, "POST", "::1", "");
    ALLOWED(c, "GET",  "foo.localhost:8080", "");
    ALLOWED(c, "GET",  "192.168.1.20:8080", "");            // an IP literal cannot be a rebinding name
    ALLOWED(c, "GET",  "", "");                              // HTTP/1.0, no Host
    REFUSED(c, "GET",  "evil.example:8080", "");             // the rebinding page
    REFUSED(c, "POST", "evil.example", "");
    REFUSED(c, "GET",  "localhost.evil.example:8080", "");   // suffix trick
    REFUSED(c, "GET",  "127.0.0.1.evil.example", "");        // not an IP literal
    REFUSED(c, "GET",  "xlocalhost", "");

    config named = c;
    named.allowed_hosts = { "llm.lan" };
    ALLOWED(named, "GET", "llm.lan:8080", "");               // opposite: allowed once named
    REFUSED(named, "GET", "other.lan:8080", "");
    config any = c;
    any.allowed_hosts = { "*" };
    ALLOWED(any, "GET", "evil.example", "");

    config wide = c;
    wide.bind_loopback = false;                              // a LAN/public server: hostnames are legitimate
    ALLOWED(wide, "GET", "evil.example", "");

    // ---- Origin on state-changing requests ----
    REFUSED(c, "POST",   "127.0.0.1:8080", "https://evil.example");
    REFUSED(c, "DELETE", "127.0.0.1:8080", "https://evil.example:8443");
    REFUSED(c, "PUT",    "127.0.0.1:8080", "null");          // sandboxed iframes / data: URLs send Origin: null
    ALLOWED(c, "POST",   "127.0.0.1:8080", "");               // curl / SDK: no Origin
    ALLOWED(c, "POST",   "127.0.0.1:8080", "http://127.0.0.1:8080");  // the server's own web UI
    ALLOWED(c, "POST",   "localhost:8080", "http://localhost:8080");
    ALLOWED(c, "POST",   "127.0.0.1:8080", "http://localhost:3000");  // a local dev front end
    ALLOWED(c, "POST",   "[::1]:8080",     "http://[::1]:3000");
    ALLOWED(c, "GET",     "127.0.0.1:8080", "https://evil.example");  // GET has no side effect here
    ALLOWED(c, "OPTIONS", "127.0.0.1:8080", "https://evil.example");  // preflight is the CORS handler's

    config cors = c;
    cors.allowed_origins = { "app.example.com", "https://other.example:9000" };
    ALLOWED(cors, "POST", "127.0.0.1:8080", "https://app.example.com");   // opposite: named origin passes
    ALLOWED(cors, "POST", "127.0.0.1:8080", "https://APP.example.com:443");
    ALLOWED(cors, "POST", "127.0.0.1:8080", "https://other.example:9000");
    REFUSED(cors, "POST", "127.0.0.1:8080", "https://evil.example");
    config open = c;
    open.cors_open = true;                                   // operator passed --cors-origins *
    ALLOWED(open, "POST", "127.0.0.1:8080", "https://evil.example");

    // a LAN server reached by its own name: same-origin POST passes, a foreign page does not
    ALLOWED(wide, "POST", "llm.lan:8080", "http://llm.lan:8080");
    REFUSED(wide, "POST", "llm.lan:8080", "http://evil.example");

    // ---- parsing helpers ----
    CHECK(hostguard::host_only("Example.COM:8080") == "example.com");
    CHECK(hostguard::host_only("[::1]:8080") == "::1");
    CHECK(hostguard::host_only("::1") == "::1");
    CHECK(hostguard::host_only("  localhost  ") == "localhost");
    CHECK(hostguard::origin_hostport("HTTPS://Evil.Example:8443/path") == "evil.example:8443");
    CHECK(hostguard::origin_hostport("null") == "null");
    CHECK(hostguard::is_ip_literal("10.0.0.1") && hostguard::is_ip_literal("fe80::1"));
    CHECK(!hostguard::is_ip_literal("10.0.0") && !hostguard::is_ip_literal("a.b.c.d") && !hostguard::is_ip_literal(""));

    if (fails) {
        std::printf("%d check(s) FAILED\n", fails);
        return 1;
    }
    std::printf("hostguard: all checks passed\n");
    return 0;
}
