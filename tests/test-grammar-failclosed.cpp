// common_grammar_failclosed (common/grammar-failclosed.h): the sticky fail-closed state of the llguidance / xgrammar samplers.
// Row S69-ENGREV-APHRODITE-LLG-FAILCLOSED-2026-10-27-01. Before, a matcher error dropped the matcher (llguidance) or kept the old state
// (xgrammar) and every later token was UNMASKED. The stub below plays a matcher that errors on its third mask; the sampler flow is the
// one common/llguidance.cpp runs (apply: failed -> EOS only; else compute the mask, an error sets the flag). Needs no model.
#include "grammar-failclosed.h"
#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

static const int   N_VOCAB = 8;
static const llama_token EOS = 7;

// stub matcher: allows tokens 0..3 only; compute_mask fails on call number `fail_on` (0 = never)
struct stub_matcher {
    int calls   = 0;
    int fail_on = 0;
    int compute() { return ++calls == fail_on ? -1 : 0; }
};

// one sampling step with the sampler flow of llama_sampler_llg_apply; returns how many candidates stay selectable
static int step(common_grammar_failclosed & fc, stub_matcher & m, std::vector<llama_token_data> & cur, bool fail_closed) {
    for (int i = 0; i < N_VOCAB; ++i) cur[i] = { i, 0.0f, 0.0f };
    llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
    if (fail_closed && fc.apply_if_failed(&arr, EOS)) {
        // already failed: EOS only
    } else if (m.compute() != 0) {
        if (fail_closed) {
            fc.fail("stub: mask computation failed");
            fc.apply_if_failed(&arr, EOS);
        }
        // fail-OPEN (the old behaviour): matcher dropped, nothing masked
    } else {
        for (int i = 4; i < N_VOCAB; ++i) cur[i].logit = -INFINITY;
    }
    int open = 0;
    for (int i = 0; i < N_VOCAB; ++i) open += std::isfinite(cur[i].logit);
    return open;
}

int main() {
    std::vector<llama_token_data> cur(N_VOCAB);

    // the defect, fail-OPEN model of the old code: tokens 4.. (every step after the failing third mask) are unconstrained
    {
        common_grammar_failclosed fc; stub_matcher m; m.fail_on = 3;
        CHECK(step(fc, m, cur, false) == 4, "step 1 constrained");
        CHECK(step(fc, m, cur, false) == 4, "step 2 constrained");
        CHECK(step(fc, m, cur, false) == 8, "OLD behaviour: step 3 (the error) leaves all 8 tokens selectable");
    }
    // fail CLOSED: from the failing mask on only EOS stays selectable, and the flag is set with the cause
    {
        common_grammar_failclosed fc; stub_matcher m; m.fail_on = 3;
        CHECK(step(fc, m, cur, true) == 4, "step 1 constrained");
        CHECK(step(fc, m, cur, true) == 4, "step 2 constrained");
        CHECK(!fc.failed, "no flag before the error");
        CHECK(step(fc, m, cur, true) == 1 && std::isfinite(cur[EOS].logit), "step 3: EOS only");
        CHECK(fc.failed && fc.error.find("stub") != std::string::npos, "flag set with the matcher's message");
        for (int s = 4; s < 8; ++s) {
            CHECK(step(fc, m, cur, true) == 1 && std::isfinite(cur[EOS].logit), "step %d stays EOS only (sticky)", s);
        }
        CHECK(m.calls == 3, "a failed sampler does not call the matcher again (%d calls)", m.calls);
        // first error wins
        fc.fail("second");
        CHECK(fc.error.find("stub") != std::string::npos, "a later error does not overwrite the first cause");
        // reset clears it
        fc.clear();
        CHECK(!fc.failed && fc.error.empty(), "clear() restores the open state");
    }
    // negative control: a matcher that never errors gives output bit-identical to the plain mask path, flag never set
    {
        common_grammar_failclosed fc; stub_matcher m; m.fail_on = 0;
        for (int s = 0; s < 6; ++s) {
            CHECK(step(fc, m, cur, true) == 4, "control step %d", s);
            for (int i = 0; i < N_VOCAB; ++i) {
                CHECK(std::isfinite(cur[i].logit) == (i < 4), "control token %d", i);
            }
        }
        CHECK(!fc.failed, "control: flag stays clear");
    }
    // a vocab without EOS: nothing can be forced, candidates untouched, flag still set
    {
        common_grammar_failclosed fc; fc.fail("x");
        for (int i = 0; i < N_VOCAB; ++i) cur[i] = { i, 0.0f, 0.0f };
        llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
        CHECK(fc.apply_if_failed(&arr, LLAMA_TOKEN_NULL), "reports failed");
        int open = 0;
        for (int i = 0; i < N_VOCAB; ++i) open += std::isfinite(cur[i].logit);
        CHECK(open == N_VOCAB, "no EOS in the vocab: candidates are not blanked");
    }

    // server finish_reason hook (row ...-FU-GRAMMAR-SERVER-FINISH-REASON): the accessors the slot polls report "not failed" on a
    // missing sampler and on a sampler that is not a grammar sampler, and never write the error out
    {
        std::string e = "untouched";
        CHECK(!common_sampler_grammar_failed(nullptr, &e) && e == "untouched", "null common_sampler: not failed, err untouched");
        CHECK(!llama_sampler_llg_failed(nullptr, &e) && e == "untouched", "null sampler (llg): not failed");
        CHECK(!llama_sampler_xgr_failed(nullptr, &e) && e == "untouched", "null sampler (xgr): not failed");
        CHECK(!common_sampler_grammar_failed(nullptr, nullptr), "null err pointer is accepted");
    }

    if (g_fails) {
        std::printf("test-grammar-failclosed: %d failure(s)\n", g_fails);
        return 1;
    }
    std::printf("test-grammar-failclosed: all pass\n");
    return 0;
}
