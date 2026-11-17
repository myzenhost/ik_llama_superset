// Fault-inject test for the /health decode-failure latch.
// Before the fix there is no latch: a compute error leaves /health at 200.
// An end-to-end GET /health after a real CUDA fault is unmeasured (needs a GGUF and a device fault).
#include "server-decode-fail-latch.h"

#include <cstdio>
#include <cstdlib>

static int n_fail = 0;
#define CHECK(c) do { if (c) { printf("  ok:   %s\n", #c); } else { printf("  FAIL: %s\n", #c); n_fail++; } } while (0)

int main() {
    server_decode_fail_latch l;
    CHECK(!l.is_failed());
    l.note_decode(1);   // KV full: request-caused
    CHECK(!l.is_failed());
    l.note_decode(-1);  // invalid batch: request-caused
    CHECK(!l.is_failed());
    l.note_decode(-3);  // compute error: latch
    CHECK(l.is_failed());
    l.note_decode(1);   // a retry that still has no slot does not clear
    CHECK(l.is_failed());
    l.note_decode(0);   // success clears
    CHECK(!l.is_failed());
    CHECK(!server_decode_fail_latch_global().is_failed());
    printf(n_fail ? "FAILED\n" : "PASSED\n");
    return n_fail ? 1 : 0;
}
