// server_drain_state: first begin() wins, state sticks, 503 text/retry hint exist.
#include "server-drain.h"

#include <cstdio>
#include <cstring>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

int main() {
    server_drain_state d;
    CHECK(!d.is_draining());
    CHECK(d.begin());   // first caller wins
    CHECK(d.is_draining());
    CHECK(!d.begin());  // second signal does not re-announce
    CHECK(d.is_draining());
    CHECK(server_drain_state::retry_after_s() >= 1);
    CHECK(std::strstr(server_drain_state::message(), "draining") != nullptr);
    printf("test-server-drain OK\n");
    return 0;
}
