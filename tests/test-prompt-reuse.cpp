#include "prompt-reuse.h"

#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
using V = std::vector<int>;

int main() {
    prompt_reuse_opts o;
    CHECK(prompt_reuse_plan({1,2,3,4}, {1,2,3,9,9}, o) == 3);       // plain divergence
    CHECK(prompt_reuse_plan({1,2,3,4}, {1,2,3,4,5,6}, o) == 3);     // cache fully matches: keep last cached token
    CHECK(prompt_reuse_plan({1,2,3,4,5}, {1,2,3}, o) == 2);         // prompt shorter: keep one prompt token
    CHECK(prompt_reuse_plan({1,2,3}, {1,2,3}, o) == 2);             // identical: still one fresh token
    CHECK(prompt_reuse_plan({7}, {7}, o) == 0);                     // deliberate deviation: never consume whole prompt
    CHECK(prompt_reuse_plan({}, {1,2}, o) == 0);
    CHECK(prompt_reuse_plan({1,2}, {}, o) == 0);
    CHECK(prompt_reuse_plan({5,2,3}, {1,2,3}, o) == 0);             // diverges at 0

    o.min_to_proceed = 4;
    CHECK(prompt_reuse_plan({1,2,3,4,5,6}, {1,2,3,9,9,9}, o) == 0); // 3 < 4: start fresh
    CHECK(prompt_reuse_plan({1,2,3,4,5,6}, {1,2,3,4,9,9}, o) == 4); // exactly at threshold

    o = {}; o.min_input_to_keep = 3;
    CHECK(prompt_reuse_plan({1,2,3,4,5,6}, {1,2,3,4,5,9}, o) == 3); // 5 > 6-3 -> 3
    CHECK(prompt_reuse_plan({1,2,3}, {1,2,3,4}, o) == 1);           // min(2,3)=2 -> 4-3=1
    CHECK(prompt_reuse_plan({1,2}, {1,2,3}, o) == 0);               // 3-3=0 -> nothing

    o = {}; o.require_full_subset = true;
    CHECK(prompt_reuse_plan({1,2,3}, {1,2,3,4,5}, o) == 3);         // cache is a full prefix: no cap
    CHECK(prompt_reuse_plan({1,2,3}, {1,2,3}, o) == 3);
    CHECK(prompt_reuse_plan({1,2,3}, {1,2,9,4,5}, o) == 0);         // mismatch: whole cache unusable
    CHECK(prompt_reuse_plan({1,2,3,4}, {1,2,3}, o) == 0);           // cache longer than prompt
    o.min_input_to_keep = 2;
    CHECK(prompt_reuse_plan({1,2,3}, {1,2,3,4}, o) == 2);           // step 4 still applies

    printf(fails ? "prompt-reuse: %d FAILED\n" : "prompt-reuse: all passed\n", fails);
    return fails ? 1 : 0;
}
