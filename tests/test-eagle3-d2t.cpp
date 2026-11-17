// EAGLE3 d2t map convention (common/eagle3-d2t.h). Model-free.
// Row S69-EAGLE3-D2T-CONVERTER-ABSOLUTE-VS-RUNTIME-DELTA-2026-10-17-01.
//
// Builds one synthetic draft->target table (a sorted subset of the target vocab, like a real reduced draft vocab),
// writes it in both on-disk conventions, and checks that each resolves to the same target ids. The FAIL-BEFORE arm
// replays the runtime's previous rule (target = i + d2t[i], unconditionally) on the absolute file and requires it to
// be wrong -- that rule is the defect this test exists for.

#include "eagle3-d2t.h"

#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

static int n_fail = 0;
static int n_check = 0;

#define CHECK(cond, ...) do { ++n_check; if (!(cond)) { ++n_fail; std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)

int main() {
    const int32_t n_vocab_tgt = 128256;  // llama-3 target vocab
    const int32_t n_draft     = 32000;   // typical EAGLE3 draft vocab

    // target ids for draft ids: a sorted random injection into the target vocab, id 0..99 kept as themselves
    // (special/byte tokens are kept by every reduced-vocab builder, which is why real delta tables start with zeros).
    std::mt19937 rng(1017);
    std::vector<int32_t> target(n_draft);
    {
        std::vector<bool> pick(n_vocab_tgt, false);
        for (int32_t i = 0; i < 100; ++i) pick[i] = true;
        int32_t have = 100;
        std::uniform_int_distribution<int32_t> u(100, n_vocab_tgt - 1);
        while (have < n_draft) { const int32_t t = u(rng); if (!pick[t]) { pick[t] = true; ++have; } }
        int32_t k = 0;
        for (int32_t t = 0; t < n_vocab_tgt; ++t) if (pick[t]) target[k++] = t;
    }

    std::vector<int32_t> absolute = target;
    std::vector<int32_t> delta(n_draft);
    for (int32_t i = 0; i < n_draft; ++i) delta[i] = target[i] - i;

    // 1. classification of each file
    CHECK(common_eagle3_d2t_classify(absolute, n_vocab_tgt) == COMMON_EAGLE3_D2T_ABSOLUTE, "absolute map not classified absolute");
    CHECK(common_eagle3_d2t_classify(delta,    n_vocab_tgt) == COMMON_EAGLE3_D2T_DELTA,    "delta map not classified delta");

    // 2. normalisation: both files resolve every draft id to the true target id
    {
        std::vector<int32_t> a = absolute, d = delta;
        CHECK(common_eagle3_d2t_normalize(a, n_vocab_tgt) == COMMON_EAGLE3_D2T_ABSOLUTE, "normalize(absolute) form");
        CHECK(common_eagle3_d2t_normalize(d, n_vocab_tgt) == COMMON_EAGLE3_D2T_DELTA,    "normalize(delta) form");
        int bad_a = 0, bad_d = 0;
        for (int32_t i = 0; i < n_draft; ++i) { bad_a += a[i] != target[i]; bad_d += d[i] != target[i]; }
        CHECK(bad_a == 0, "absolute file: %d of %d draft ids map to the wrong target id", bad_a, n_draft);
        CHECK(bad_d == 0, "delta file: %d of %d draft ids map to the wrong target id", bad_d, n_draft);
    }

    // 3. FAIL-BEFORE: the previous runtime rule (i + d2t[i], always) on the absolute file is wrong for nearly every id
    {
        int wrong = 0, out_of_range = 0;
        for (int32_t i = 0; i < n_draft; ++i) {
            const int64_t old = (int64_t) i + absolute[i];
            wrong        += old != target[i];
            out_of_range += old >= n_vocab_tgt;
        }
        std::printf("old delta-only rule on the absolute file: %d of %d ids wrong, %d past the vocab\n", wrong, n_draft, out_of_range);
        CHECK(wrong == n_draft - 1, "expected every id but 0 wrong under the old rule, got %d", wrong);
    }

    // 4. identity delta table (draft vocab == target vocab, all zeros) is a delta map, not an absolute one
    {
        std::vector<int32_t> z(1000, 0);
        CHECK(common_eagle3_d2t_classify(z, 1000) == COMMON_EAGLE3_D2T_DELTA, "all-zero table should read as delta identity");
        common_eagle3_d2t_normalize(z, 1000);
        int bad = 0; for (int32_t i = 0; i < 1000; ++i) bad += z[i] != i;
        CHECK(bad == 0, "all-zero delta table did not normalise to identity (%d bad)", bad);
    }

    // 5. a map that is an injection in BOTH readings resolves as absolute (upstream's convention)
    {
        std::vector<int32_t> both = {0, 2, 4, 6};          // absolute: 0,2,4,6; delta: 0,3,6,9 -- both valid in 16
        CHECK(common_eagle3_d2t_classify(both, 16) == COMMON_EAGLE3_D2T_ABSOLUTE, "ambiguous map should prefer absolute");
    }

    // 6. invalid maps: duplicates in both readings, out-of-range, empty, non-positive vocab
    {
        std::vector<int32_t> dup = {5, 5, 4};               // absolute dup 5; delta 5,6,6 dup
        CHECK(common_eagle3_d2t_classify(dup, 16) == COMMON_EAGLE3_D2T_INVALID, "duplicate map should be invalid");
        std::vector<int32_t> oor = {20, 30};                // out of a 16-token vocab either way
        CHECK(common_eagle3_d2t_classify(oor, 16) == COMMON_EAGLE3_D2T_INVALID, "out-of-range map should be invalid");
        std::vector<int32_t> neg = {-1, -1};                // absolute negative; delta -1,0 -> -1 negative
        CHECK(common_eagle3_d2t_classify(neg, 16) == COMMON_EAGLE3_D2T_INVALID, "negative map should be invalid");
        std::vector<int32_t> keep = dup;
        CHECK(common_eagle3_d2t_normalize(keep, 16) == COMMON_EAGLE3_D2T_INVALID && keep == dup, "invalid map must be left untouched");
        CHECK(common_eagle3_d2t_classify({}, 16) == COMMON_EAGLE3_D2T_INVALID, "empty map should be invalid");
        CHECK(common_eagle3_d2t_classify(absolute, 0) == COMMON_EAGLE3_D2T_INVALID, "vocab 0 should be invalid");
    }

    // 7. a real delta table from the fork's #29 gguf shape: zeros, then growing offsets (d2t[1000] = 131 measured)
    {
        std::vector<int32_t> real_shape = delta;
        CHECK(real_shape[50] == 0 && real_shape[n_draft - 1] > 0, "synthetic delta table lost its real-file shape");
    }

    std::printf("test-eagle3-d2t: %d checks, %d failed\n", n_check, n_fail);
    return n_fail == 0 ? 0 : 1;
}
