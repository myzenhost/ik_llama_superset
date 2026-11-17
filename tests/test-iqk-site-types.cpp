// ggml_iqk_supports_type(type, site): pin each iqk call site's weight-type set.
//
// Row S69-FORK-CARRY-IQK-SUPPORTS-TYPE-PREDICATE-01. Before 2026-10-18 each site carried its own
// `type != ...` chain and the chains had drifted apart (dense: none, MoE: 2 types, CPU fused
// up/gate: 4, the ggml-layer op guard: 3, the graph FFN fusion: 7, the graph MoE fusion: 3).
// The drift is deliberate -- kernel coverage really does differ per site -- so the sets are
// pinned here, one expected set per site, every type id checked, and any change to a set has to
// change this file too.
//
// Control: the comparator is run once against a deliberately wrong expected set (one type
// dropped) and must report exactly one mismatch; an out-of-range site must take nothing.

#include "ggml.h"

#include <cstdio>
#include <set>
#include <vector>

struct site_case {
    const char *          name;
    ggml_iqk_site         site;
    std::set<int>         excluded;
};

static int mismatches(const site_case & c, bool verbose) {
    int bad = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const bool want = c.excluded.count(t) == 0;
        const bool got  = ggml_iqk_supports_type((ggml_type) t, c.site);
        if (want != got) {
            ++bad;
            if (verbose) {
                const char * nm = ggml_type_name((ggml_type) t);
                std::printf("  MISMATCH %s type %d (%s): want %d got %d\n", c.name, t, nm ? nm : "?", want, got);
            }
        }
    }
    return bad;
}

int main() {
    // Gyro trellis types (ik df93e4cc9 #2812): dense GEMM (TQ2_T) / decode only; every non-dense
    // site declines them (ggml_is_tq_trellis_type), so each non-dense set below carries these 6 ids.
    const std::set<int> trellis = { GGML_TYPE_TQK6, GGML_TYPE_TQK7, GGML_TYPE_TQ2_T,
                                    GGML_TYPE_TQP5, GGML_TYPE_TQP6, GGML_TYPE_TQP7 };
    auto with_trellis = [&](std::set<int> s) { s.insert(trellis.begin(), trellis.end()); return s; };
    const std::set<int> tq_g128 = with_trellis({ GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_Q2_0_G128 });

    std::vector<site_case> cases = {
        { "dense",           GGML_IQK_SITE_DENSE,             {} },
        { "moe",             GGML_IQK_SITE_MOE,               with_trellis({ GGML_TYPE_Q2_0_G128, GGML_TYPE_Q2_0 }) },
        { "cpu_up_gate",     GGML_IQK_SITE_CPU_UP_GATE,       with_trellis({ GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_Q2_0_G128, GGML_TYPE_Q2_0 }) },
        { "up_gate_op",      GGML_IQK_SITE_UP_GATE_OP,        tq_g128 },
        { "graph_ffn",       GGML_IQK_SITE_GRAPH_FFN_UP_GATE, with_trellis({ GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_Q2_0_G128,
                                                                 GGML_TYPE_PQ2_0, GGML_TYPE_PQ2_0_R8,
                                                                 GGML_TYPE_PTQ1_0, GGML_TYPE_PTQ1_0_R8 }) },
        { "graph_moe",       GGML_IQK_SITE_GRAPH_MOE_UP_GATE, tq_g128 },
    };

    int failed = 0;
    for (const auto & c : cases) {
        const int bad = mismatches(c, true);
        std::printf("%-12s excluded %zu of %d type ids: %s\n", c.name, c.excluded.size(), (int) GGML_TYPE_COUNT,
                    bad == 0 ? "ok" : "FAIL");
        failed += bad != 0;
    }

    // every site in the enum is covered by a case above
    if ((int) cases.size() != (int) GGML_IQK_SITE_COUNT) {
        std::printf("FAIL: %zu cases for %d sites\n", cases.size(), (int) GGML_IQK_SITE_COUNT);
        ++failed;
    }

    // control 1: the comparator sees a wrong expected set (Q2_0 dropped from the MoE set)
    site_case wrong = cases[1];
    wrong.excluded.erase(GGML_TYPE_Q2_0);
    const int wrong_bad = mismatches(wrong, false);
    std::printf("control: wrong MoE set -> %d mismatch(es) (must be 1)\n", wrong_bad);
    if (wrong_bad != 1) ++failed;

    // control 2: an unknown site takes nothing
    const bool unknown = ggml_iqk_supports_type(GGML_TYPE_F32, GGML_IQK_SITE_COUNT);
    std::printf("control: unknown site takes F32 -> %d (must be 0)\n", (int) unknown);
    if (unknown) ++failed;

    if (failed == 0) {
        std::printf("OK: every iqk site set matches\n");
        return 0;
    }
    std::printf("FAILED: %d check(s)\n", failed);
    return 1;
}
