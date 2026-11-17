#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

// THIS HEADER USED TO `#include "common.h" // for llama_tokens`, AND THAT WAS A CYCLE.
//
//     common.h:19            #include "sampling.h"
//     sampling.h:5           #include "reasoning-budget.h"
//     reasoning-budget.h:4   #include "common.h"      <- no-op, common.h is already in progress
//
// `#pragma once` makes the back edge silently do nothing, so this header was compiled with only
// the first 19 lines of common.h visible -- and `using llama_tokens` lives at common.h:77, well
// after that point. The result was 80 of the 434 error lines in the `common` target:
//     reasoning-budget.h:58: error: 'llama_tokens' was not declared in this scope
//     sampling.h:205:        error: 'llama_tokens' was not declared in this scope
// each repeated once per translation unit, plus the invalid-template-argument errors cascading
// from them. The include looked correct and the comment beside it said exactly why it was there;
// only the ORDER made it useless.
//
// Declaring the alias here makes this header self-contained, which is what breaks the cycle
// instead of reordering it and waiting for the next include to re-create it. Repeating an
// alias-declaration with the identical type is well-formed, and common.h itself already declares
// this one twice (at :77 and :95, a separate merge artifact).
using llama_tokens = std::vector<llama_token>;

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,         // waiting for start sequence
    REASONING_BUDGET_COUNTING,     // counting down tokens
    REASONING_BUDGET_FORCING,      // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8, // budget exhausted, waiting for UTF-8 completion
    REASONING_BUDGET_DONE,         // passthrough forever
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:         passthrough, watching for a start sequence
//   COUNTING:     counting down remaining tokens, watching for a natural end sequence
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough forever
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_tokens   - token sequence that activates counting
//   end_seqs       - token sequences, any of which naturally deactivates (widened from a single
//                    sequence to a list: decision #5 Part C, MEASURED against
//                    server-schema.cpp's .push_back(tokenize(...)) usage — see
//                    common/sampling.h's reasoning_budget_end field comment)
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   initial_state  - initial state
//
// NOTE ON TYPE: this fork's opaque common_reasoning_budget_ctx* is the PRIMARY type (per
// S69-DECISION-5-MADE-OURS-STRUCT-PRIMARY-2026-09-07-01 — common/sampling.h's `rbudget` field is
// typed common_reasoning_budget_ctx*, not llama_sampler*, and common/sampling.cpp already calls
// these functions against that type). unsloth/master's incoming design instead wraps this as a
// registered llama_sampler chain member (struct llama_sampler* return type, llama_sampler_i vtable
// in reasoning-budget.cpp). Both designs are complete; ours' is kept primary so as not to reopen
// the already-staged common.h/sampling.h/sampling.cpp decision. get_end_match()/force() are
// unsloth's capabilities, ported onto ours' ctx type rather than dropped, so no capability is lost.
//
// RESEARCH-DONE: read common/sampling.h (rbudget field + its decision-#5-Part-C comment),
// common/sampling.cpp (every rbudget/common_reasoning_budget_* call site, lines 29-1463), and
// tests/test-reasoning-budget.cpp in full before choosing this shape. That test file called
// common_reasoning_budget_init/_force/_get_end_match against the llama_sampler* signature with a
// plural start too. DONE 2026-10-05 (integ/absent-branches-1005): the test is rewritten for the
// ctx API (apply/accept/clone/free) and uses common_reasoning_budget_init_seqs for the plural-start
// cases; tracked under task S69-DECISION-5-ADAPTER-LOGIT-BIAS-REASONING-END-2026-09-07-01.

struct common_reasoning_budget_ctx * common_reasoning_budget_init(
    const struct llama_vocab * vocab,
    const std::vector<llama_token> & start_tokens,
    const std::vector<llama_tokens> & end_seqs,
    const std::vector<llama_token> & forced_tokens,
    int32_t                          budget,
    common_reasoning_budget_state    initial_state = REASONING_BUDGET_IDLE);

// Same as common_reasoning_budget_init, but ANY of several start sequences activates (and re-arms) the budget.
struct common_reasoning_budget_ctx * common_reasoning_budget_init_seqs(
    const struct llama_vocab * vocab,
    const std::vector<llama_tokens> & start_seqs,
    const std::vector<llama_tokens> & end_seqs,
    const std::vector<llama_token> & forced_tokens,
    int32_t                          budget,
    common_reasoning_budget_state    initial_state = REASONING_BUDGET_IDLE);

common_reasoning_budget_state common_reasoning_budget_get_state(const common_reasoning_budget_ctx * smpl);

// The end sequence that transitioned the ctx to DONE, or nullptr if none was recorded.
// Cleared when a new start sequence re-arms it. (Ported from unsloth/master.)
const llama_tokens * common_reasoning_budget_get_end_match(const common_reasoning_budget_ctx * smpl);

// Manually transition the reasoning budget ctx into the FORCING state.
// Returns true if the transition occurred. (Ported from unsloth/master.)
bool common_reasoning_budget_force(common_reasoning_budget_ctx * smpl);

// EOG-inside-reasoning repair (row S69-ENGREV-BEELLAMA-EOG-IN-REASONING-FORCE-CLOSE-2026-10-27-01).
// A reasoning model can sample an end-of-generation token while a <think> block is still open; the caller then
// stops with empty content. Call this with the token the sampler just chose: when `token_is_eog` is true and the ctx
// is COUNTING or WAITING_UTF8 (reasoning open) and the ctx has forced tokens, the ctx moves to FORCING from position 0
// and this returns true -- the caller must then re-apply the ctx to fresh candidates and draw again, which yields the
// first token of the end-of-reasoning sequence instead of EOG. Returns false (ctx untouched) in every other case:
// not EOG, IDLE/FORCING/DONE, no forced tokens, null ctx.
bool common_reasoning_budget_repair_eog(common_reasoning_budget_ctx * smpl, bool token_is_eog);

// Feed one generated/accepted token into the state machine.
void common_reasoning_budget_accept(common_reasoning_budget_ctx * smpl, llama_token token);

// Apply budget-driven logit masking/forcing to the candidate array for the current step.
void common_reasoning_budget_apply(common_reasoning_budget_ctx * smpl, llama_token_data_array * cur_p);

// Deep-clone a reasoning budget ctx (used by common_sampler_copy/common_sampler_clone).
struct common_reasoning_budget_ctx * common_reasoning_budget_clone(const struct common_reasoning_budget_ctx * smpl);

// Free a reasoning budget ctx created by common_reasoning_budget_init.
void common_reasoning_budget_free(struct common_reasoning_budget_ctx * smpl);
