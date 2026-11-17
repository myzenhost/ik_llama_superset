// server_truncate_plan_for: fits -> untouched; over -> head + tail within the budget; n_keep bounded; n_predict reserved.
#include "server-truncate.h"

#include <cstdio>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

int main() {
    // fits: 100 tokens in 4096
    CHECK(!server_truncate_plan_for(100, 4096, -1, 0).truncate);
    // exactly the budget (n_ctx - 1) fits, one more does not
    CHECK(!server_truncate_plan_for(4095, 4096, -1, 0).truncate);
    {
        auto p = server_truncate_plan_for(4096, 4096, -1, 0);
        CHECK(p.truncate && p.keep_head == 0 && p.keep_tail == 4095 && p.n_dropped == 1);
    }
    // keep a 64-token system prompt, drop the middle of a 10000-token prompt
    {
        auto p = server_truncate_plan_for(10000, 4096, -1, 64);
        CHECK(p.truncate && p.keep_head == 64 && p.keep_head + p.keep_tail == 4095);
        CHECK(p.keep_head + p.keep_tail + p.n_dropped == 10000);
    }
    // the reply gets room: n_predict 512 -> prompt budget 4096 - 1 - 512
    {
        auto p = server_truncate_plan_for(10000, 4096, 512, 0);
        CHECK(p.keep_tail == 4096 - 1 - 512);
    }
    // n_predict larger than half the context is capped to half
    {
        auto p = server_truncate_plan_for(10000, 4096, 100000, 0);
        CHECK(p.keep_tail == 4096 - 1 - 2048);
    }
    // an n_keep bigger than the budget cannot starve the tail
    {
        auto p = server_truncate_plan_for(10000, 4096, -1, 100000);
        CHECK(p.keep_head == 4095 / 2 && p.keep_tail > 0);
    }
    // degenerate context
    CHECK(!server_truncate_plan_for(10, 1, -1, 0).truncate);
    printf("test-server-truncate OK\n");
    return 0;
}
