// test-ckpt-dominance: common_ckpt_dominated() (--ctx-ckpt-dominance). No model. Prints "N checks, M failed", non-zero on failure.
// Row TRNRT-ADAPT-CKPT-DOMINATED-FREE (S69-ENGREV-TRNRT, 2026-11-13). Each rule has a negative control.
#include "common.h"
#include "ckpt-dominance.h"

#include <cstdio>
#include <vector>

static int g_checks = 0;
static int g_fails  = 0;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: %s\n", what); } } while (0)

using E = common_ckpt_dom_entry;
using V = std::vector<size_t>;

int main() {
    // flag defaults to the current behaviour
    { common_params p; CHECK(!p.ctx_checkpoint_dominance, "default: dominance off"); }

    // nothing to free with 0 or 1 checkpoint
    CHECK(common_ckpt_dominated({}).empty(), "empty list frees nothing");
    CHECK(common_ckpt_dominated({E{100}}).empty(), "single checkpoint frees nothing");

    // unbranched path, no anchors: only the deepest survives
    CHECK((common_ckpt_dominated({E{100}, E{200}, E{300}}) == V{0, 1}), "older ckpts on an unbranched path are freed, deepest kept");

    // the newest prompt-end anchor is kept, older dominated ones go (incl. an older anchor)
    {
        const auto f = common_ckpt_dominated({E{100, true}, E{200}, E{300, true}, E{400}});
        CHECK((f == V{0, 1}), "newest anchor (300) and deepest (400) kept; older anchor (100) freed");
    }
    // deepest is itself the anchor: no other anchor kept
    CHECK((common_ckpt_dominated({E{100, true}, E{200}, E{300, true}}) == V{0, 1}), "deepest anchor: older anchor freed too");

    // branched path: the branch-point checkpoint and everything below it are kept
    {
        const auto f = common_ckpt_dominated({E{100}, E{200, false, true}, E{300}, E{400}});
        CHECK((f == V{2}), "walk stops at the branch point: 200 and 100 kept, 300 freed, 400 deepest");
    }
    CHECK(common_ckpt_dominated({E{100}, E{200, false, true}, E{300}}).empty(), "branch point directly below the deepest: nothing freed");
    CHECK(common_ckpt_dominated({E{100, false, true}, E{200}}).empty(), "branch ckpt directly below the deepest: both kept");

    // order of the list does not matter (sorted by n_tokens)
    CHECK((common_ckpt_dominated({E{300}, E{100}, E{200}}) == V{1, 2}), "unsorted input: indices refer to the input positions");

    // negative control: a rule that frees everything but the deepest would also free the branch point
    CHECK(common_ckpt_dominated({E{100}, E{200, false, true}, E{300}, E{400}}).size() != 2, "control: branch point is not freed");

    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
