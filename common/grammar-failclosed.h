#pragma once

// Fail-CLOSED state for the grammar samplers (common/llguidance.cpp, common/xgrammar.cpp).
// Row S69-ENGREV-APHRODITE-LLG-FAILCLOSED-2026-10-27-01 (spec from aphrodite-engine / rtp-llm GrammarLogitsProcessor; AGPL/Apache sources
// read, nothing copied). Before: an llguidance matcher error (consume or compute_mask) dropped the matcher and every later token was
// UNMASKED (one MEASURED real case, SPM forced bytes 2026-09-28: constrained JSON turned into free text); an xgrammar rejected token was
// logged and the matcher kept its old state (silent drift). After: the first error sets a sticky flag; from then on only the end-of-
// generation token stays selectable, so the request ends instead of continuing unconstrained.
// Header-only and free of llguidance / xgrammar so a CPU unit test covers the state machine (tests/test-grammar-failclosed.cpp).

#include "llama.h"

#include <cmath>
#include <string>

struct common_grammar_failclosed {
    bool        failed = false;
    std::string error;

    // first error wins; later calls never overwrite the cause
    void fail(const char * what) {
        if (!failed) {
            failed = true;
            error  = what ? what : "grammar error";
        }
    }

    void clear() {
        failed = false;
        error.clear();
    }

    // Constrain cur_p to the end-of-generation token. Returns true when the sampler is in the failed state (the caller must then
    // skip its normal mask). eos == LLAMA_TOKEN_NULL (a vocab without EOS/EOT): nothing can be forced, the candidates are left as
    // they are; the flag stays set so the owner of the request can still end it with an explicit error.
    bool apply_if_failed(llama_token_data_array * cur_p, llama_token eos) const {
        if (!failed) {
            return false;
        }
        if (eos == LLAMA_TOKEN_NULL) {
            return true;
        }
        for (size_t i = 0; i < cur_p->size; ++i) {
            if (cur_p->data[i].id != eos) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
        return true;
    }
};
