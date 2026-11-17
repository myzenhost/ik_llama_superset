// The SWA state-reuse threshold (row S69-LLAMACPP-PICK-SWA-REUSE-THRESHOLD-OFF-BY-TWO-2026-10-18-01), model-free.
// A state is reused when pos_min < server_swa_reuse_pos_min_thold(pos_next, has_new_tokens, n_swa).
//
// The boundary cases: with a window of n_swa the first evaluated token at pos_eval needs positions
// pos_eval - n_swa + 1 .. pos_eval - 1 (and pos_eval itself is recomputed). So a state starting at
// pos_eval - n_swa + 1 (exactly the window, what a trimmed SWA cache saves) must be accepted, and a state starting
// one later must be refused. Before the fix the threshold was pos_eval - n_swa and the exact-window state was refused.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "server-swa-reuse.h"

#include <cstdio>

static int n_fail = 0;

static void expect(bool ok, const char * what) {
    printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        n_fail++;
    }
}

static bool reused(llama_pos pos_min, llama_pos pos_next, bool has_new_tokens, int32_t n_swa) {
    return pos_min < server_swa_reuse_pos_min_thold(pos_next, has_new_tokens, n_swa);
}

int main() {
    const int32_t n_swa = 64;

    // resume with new tokens: pos_eval = pos_next = 320, the window needs 257..319
    expect( reused(257, 320, true,  n_swa), "new tokens: a state holding exactly the window (pos_min 257) is reused");
    expect( reused(256, 320, true,  n_swa), "new tokens: a state with one extra cell (pos_min 256) is reused");
    expect(!reused(258, 320, true,  n_swa), "new tokens: a state missing the oldest window cell (pos_min 258) is refused");

    // same prompt again: the last token is evaluated again, pos_eval = 319, the window needs 256..318
    expect( reused(256, 320, false, n_swa), "re-eval last: a state holding exactly the window (pos_min 256) is reused");
    expect(!reused(257, 320, false, n_swa), "re-eval last: a state missing the oldest window cell (pos_min 257) is refused");

    // a full (untrimmed) state is always reused
    expect( reused(0, 320, true,  n_swa), "a state from position 0 is reused");

    // no SWA: every position before pos_eval is needed, so only pos_min < pos_eval passes (unchanged by the fix)
    expect( reused(0,   320, true,  0), "no SWA: a state from position 0 is reused");
    expect( reused(319, 320, true,  0), "no SWA: pos_min 319 < pos_eval 320 is reused");
    expect(!reused(320, 320, true,  0), "no SWA: pos_min 320 is refused");
    expect(!reused(319, 320, false, 0), "no SWA, re-eval last: pos_min 319 is refused");

    // a short prompt never yields a negative threshold
    expect(server_swa_reuse_pos_min_thold(10, true, n_swa) == 0, "the threshold is clamped at 0");

    // --swa-compress rewind floor (pos_base_swa 192 + window 64 = 256), state ends at 401 (pos_end)
    expect( server_swa_reuse_rewind_ok(255, true,    0, 401), "no compaction (floor 0): any rewind is fine");
    expect(!server_swa_reuse_rewind_ok(255, true,  256, 401), "compacted: p0 255 below the floor 256 is refused");
    expect( server_swa_reuse_rewind_ok(256, true,  256, 401), "compacted: p0 256 at the floor is fine");
    expect(!server_swa_reuse_rewind_ok(256, false, 256, 401), "compacted, re-eval last: p0 255 is refused");
    expect( server_swa_reuse_rewind_ok(257, false, 256, 401), "compacted, re-eval last: p0 256 is fine");
    expect( server_swa_reuse_rewind_ok(0,   true,  256, 401), "compacted: a full reset (p0 0) is fine");
    expect( server_swa_reuse_rewind_ok(250, true,  256, 250), "compacted: p0 at the stored end removes nothing");
    // the case the threshold alone let through: pos_min 192 < thold(255) = 193, but the window cannot go back to 255
    expect(reused(192, 255, true, n_swa) && !server_swa_reuse_rewind_ok(255, true, 256, 401),
           "pos_min 192 passes the threshold at pos_next 255 but the compacted window cannot be rewound there");

    printf("%s (%d failed)\n", n_fail == 0 ? "PASS" : "FAILED", n_fail);
    return n_fail == 0 ? 0 : 1;
}
