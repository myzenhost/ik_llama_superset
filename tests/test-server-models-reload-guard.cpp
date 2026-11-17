// Router reload empty-preset guard (tools/server/server-models-reload-guard.h, --reload-allow-empty).
// Row S69-ENGREV-TFSERVING-RELOAD-EMPTY-GUARD-2026-10-27-01. tensorflow/serving is Apache-2.0; design only, no code copied.
// Needs no model. The three cases of the row's served test, as the decision the router takes in load_models():
//   1. ini overwritten with an empty file while models run      -> refuse (both stay running)
//   2. same with --reload-allow-empty                           -> reconcile (old behaviour)
//   3. ini with one section removed (set not empty)             -> reconcile (only that model unloads)

#include "server-models-reload-guard.h"

#include <cstdio>

static int g_fail = 0;

#define CHECK(cond) do {                                        \
    if (!(cond)) {                                              \
        printf("  FAIL: %s\n", #cond);                          \
        g_fail++;                                               \
    } else {                                                    \
        printf("  ok:   %s\n", #cond);                          \
    }                                                           \
} while (0)

int main() {
    CHECK( server_models_reload_refuse_empty(0, 2, false)); // empty source, two running: refuse
    CHECK( server_models_reload_refuse_empty(0, 1, false));
    CHECK(!server_models_reload_refuse_empty(0, 2, true));  // operator override: old behaviour
    CHECK(!server_models_reload_refuse_empty(1, 2, false)); // one section removed, not empty: normal reconcile
    CHECK(!server_models_reload_refuse_empty(3, 3, false)); // nothing removed
    CHECK(!server_models_reload_refuse_empty(0, 0, false)); // empty source, nothing running: nothing to protect

    printf("%s\n", g_fail ? "FAILED" : "PASSED");
    return g_fail ? 1 : 0;
}
