#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Multi-sequence matcher: ANY of `seqs` (an Aho-Corasick trie over them) can trigger a match.
// Ported from unsloth/master (single-sequence prefix matcher there was ours'; theirs' Aho-Corasick
// design is a strict superset — N==1 degenerates to the old behavior — and is required by
// common/sampling.h's reasoning_budget_end field, which decision #5 Part C widened to
// std::vector<llama_tokens>).
struct token_matcher {
    std::vector<llama_tokens> seqs;
    common_aho_corasick ac;
    size_t state = 0;

    token_matcher(const std::vector<llama_tokens> & seqs) : seqs(collect(seqs)), ac(build_trie(this->seqs)) {}

    static std::vector<llama_tokens> collect(const std::vector<llama_tokens> & seqs) {
        std::vector<llama_tokens> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    static common_trie build_trie(const std::vector<llama_tokens> & seqs) {
        common_trie t;
        for (const auto & seq : seqs) {
            t.insert(std::vector<uint32_t>(seq.begin(), seq.end()));
        }
        return t;
    }

    // returns the index into seqs of the longest sequence ending at this token, or -1
    int32_t advance(llama_token token) {
        state = ac.next(state, (uint32_t) token);
        const int32_t p = ac.match_pattern(state);
        if (p >= 0) {
            state = 0;
        }
        return p;
    }

    void reset() { state = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    llama_tokens forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    int32_t end_match;        // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none
};

// NOTE: this fork keeps common_reasoning_budget_ctx as a plain heap struct, NOT registered as a
// llama_sampler chain member (no llama_sampler_i vtable) — see S69-DECISION-5-MADE-OURS-STRUCT-
// PRIMARY-2026-09-07-01 and common/reasoning-budget.h. common/sampling.h's `rbudget` field and
// common/sampling.cpp's call sites all use this ctx type directly.
// RESEARCH-DONE: read common/sampling.h, common/sampling.cpp (every rbudget call site), and the
// task-store decision rows S69-DECISION-5-MADE-OURS-STRUCT-PRIMARY-2026-09-07-01 /
// S69-DECISION-5-COMMON-SAMPLER-DESIGN-DOSSIER-2026-09-07-01 in full before choosing this shape —
// the llama_sampler-chain wrapper (theirs) is not dropped for lack of looking, it is superseded by
// an already-made, evidence-based architecture decision that already has staged, dependent code.

void common_reasoning_budget_accept(common_reasoning_budget_ctx * ctx, llama_token token) {
    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            if (ctx->vocab != nullptr) {
                const std::string piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                if (ctx->remaining <= 0) {
                    if (utf8_complete) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, waiting for UTF-8 completion\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // track the end sequence within forced_tokens so it is also reported on DONE
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->force_pos++;
            // an end tag that is already in the prompt (e.g. "<think></think>") also ends forcing
            if (match >= 0 || ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "forced sequence complete, done\n");
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                ctx->end_match = -1;
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
    }
}

void common_reasoning_budget_apply(common_reasoning_budget_ctx * ctx, llama_token_data_array * cur_p) {
    if (!ctx) {
        return;
    }
    if (ctx->state != REASONING_BUDGET_FORCING) {
        // passthrough — don't modify logits
        return;
    }

    if (ctx->force_pos >= ctx->forced_tokens.size()) {
        return;
    }

    const llama_token forced = ctx->forced_tokens[ctx->force_pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(common_reasoning_budget_ctx * ctx) {
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
}

struct common_reasoning_budget_ctx * common_reasoning_budget_clone(const struct common_reasoning_budget_ctx * ctx) {
    return new common_reasoning_budget_ctx(*ctx);
}

void common_reasoning_budget_free(struct common_reasoning_budget_ctx * ctx) {
    delete ctx;
}

static struct common_reasoning_budget_ctx * common_reasoning_budget_init_state(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const std::vector<llama_token>  & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    return new common_reasoning_budget_ctx{
        /* .vocab         = */ vocab,
        /* .start_matcher = */ token_matcher(start_seqs),
        /* .end_matcher   = */ token_matcher(end_seqs),
        /* .forced_tokens = */ forced_tokens,
        /* .budget        = */ budget,
        /* .remaining     = */ budget,
        /* .state         = */ initial_state,
        /* .force_pos     = */ 0,
        /* .end_match     = */ -1,
    };
}

struct common_reasoning_budget_ctx * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_token>  & start_tokens,
        const std::vector<llama_tokens> & end_seqs,
        const std::vector<llama_token>  & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    return common_reasoning_budget_init_state(vocab, std::vector<llama_tokens>{ start_tokens }, end_seqs, forced_tokens, budget, initial_state);
}

// Plural-start form (any of start_seqs re-arms/activates the budget; empty sequences are dropped by token_matcher).
// Same ordering as upstream's single entry point, whose `start` parameter is a list. The single-start form above keeps
// its signature because common/sampling.cpp is its only production caller and carries one start tag.
struct common_reasoning_budget_ctx * common_reasoning_budget_init_seqs(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const std::vector<llama_token>  & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const common_reasoning_budget_ctx * ctx) {
    if (!ctx) {
        return REASONING_BUDGET_IDLE;
    }
    return ctx->state;
}

const llama_tokens * common_reasoning_budget_get_end_match(const common_reasoning_budget_ctx * ctx) {
    if (!ctx) {
        return nullptr;
    }

    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_repair_eog(common_reasoning_budget_ctx * ctx, bool token_is_eog) {
    if (!ctx || !token_is_eog || ctx->forced_tokens.empty()) {
        return false;
    }
    // reasoning is open only while counting down or waiting for a UTF-8 boundary; IDLE (no tag seen), FORCING (already closing)
    // and DONE (block closed, EOG is the real end) are left alone
    if (ctx->state != REASONING_BUDGET_COUNTING && ctx->state != REASONING_BUDGET_WAITING_UTF8) {
        return false;
    }
    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "EOG sampled inside reasoning, forcing the end sequence\n");
    return true;
}

bool common_reasoning_budget_force(common_reasoning_budget_ctx * ctx) {
    if (!ctx) {
        return false;
    }

    // only a ctx that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
