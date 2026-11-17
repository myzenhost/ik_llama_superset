// Server-side URL fetches (--media remote URLs) must not reach private / loopback / link-local / metadata hosts (SSRF).
// Pins the literal-address policy in common/http.h: common_http_ip_is_private() and common_http_host_is_private().
// No network is used: only IP literals and the localhost names, which are decided without a DNS lookup.

#include "http.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define REQUIRE_PRIV(h, want) do {                                                             \
    const bool got = common_http_host_is_private(h);                                           \
    if (got != (want)) {                                                                       \
        fprintf(stderr, "%s:%d: host '%s' private=%d, expected %d\n", __FILE__, __LINE__, h, got, (int) (want)); \
        std::abort();                                                                          \
    }                                                                                          \
} while (0)

int main() {
    // private / loopback / link-local / metadata: all blocked
    REQUIRE_PRIV("127.0.0.1",       true);
    REQUIRE_PRIV("127.255.0.9",     true);
    REQUIRE_PRIV("10.0.0.5",        true);
    REQUIRE_PRIV("172.16.0.1",      true);
    REQUIRE_PRIV("172.31.255.255",  true);
    REQUIRE_PRIV("192.168.1.10",    true);
    REQUIRE_PRIV("169.254.169.254", true);  // cloud metadata
    REQUIRE_PRIV("100.64.0.1",      true);  // carrier-grade NAT
    REQUIRE_PRIV("0.0.0.0",         true);
    REQUIRE_PRIV("localhost",       true);
    REQUIRE_PRIV("LOCALHOST",       true);
    REQUIRE_PRIV("app.localhost",   true);
    REQUIRE_PRIV("::1",             true);
    REQUIRE_PRIV("::",              true);
    REQUIRE_PRIV("fe80::1",         true);
    REQUIRE_PRIV("fd00::1",         true);
    REQUIRE_PRIV("fc00::5",         true);
    REQUIRE_PRIV("::ffff:127.0.0.1",   true);  // v4-mapped loopback
    REQUIRE_PRIV("::ffff:169.254.169.254", true);

    // public: allowed
    REQUIRE_PRIV("8.8.8.8",         false);
    REQUIRE_PRIV("1.1.1.1",         false);
    REQUIRE_PRIV("172.32.0.1",      false); // just outside 172.16/12
    REQUIRE_PRIV("172.15.255.255",  false);
    REQUIRE_PRIV("100.128.0.1",     false); // just outside 100.64/10
    REQUIRE_PRIV("2606:4700:4700::1111", false);
    REQUIRE_PRIV("::ffff:8.8.8.8",  false);

    printf("test-http-private-host: OK\n");
    return 0;
}
